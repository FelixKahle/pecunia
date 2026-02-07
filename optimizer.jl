# Copyright (c) 2026 Felix Kahle.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

module Pecunia

using JuMP
using Gurobi
using YFinance
using DataFrames
using Statistics
using LinearAlgebra
using Dates
using ProgressMeter
using Printf
using CSV
using CairoMakie
using ColorSchemes

# ==============================================================================
#  1. DOMAIN MODELS & CONFIGURATION
# ==============================================================================

"""
    AssetCategory

Enum representing the classification of the financial instrument.
"""
@enum AssetCategory Stock Bond

"""
    AssetDefinition

A data structure representing a single investable asset.
"""
struct AssetDefinition
    """ The unique identifier for the asset (e.g., Ticker Symbol like 'AAPL'). """
    ticker_symbol::String

    """ The category of the asset (Stock or Bond). """
    category::AssetCategory

    """ Human-readable name of the asset. """
    display_name::String

    """ Flag indicating if data is manually provided (True) or fetched via API (False). """
    is_manual_entry::Bool

    """ Fixed nominal yield for manual assets (e.g., bonds). 0.0 for stocks. """
    fixed_nominal_yield::Float64

    """ Annualized standard deviation of returns (volatility). """
    annualized_risk_sigma::Float64

    """ Optional hard constraint: Minimum percentage allocation (0.0 to 1.0). """
    minimum_allocation_weight::Union{Float64,Nothing}

    """ Optional hard constraint: Maximum percentage allocation (0.0 to 1.0). """
    maximum_allocation_weight::Union{Float64,Nothing}
end

"""
    PortfolioConfiguration

Configuration object containing all scalar parameters for the optimization engine,
economic assumptions, and data processing settings.
"""
struct PortfolioConfiguration
    # --- Localization ---
    """ ISO currency code (e.g., 'EUR'). """
    base_currency_code::String
    """ Currency symbol for display (e.g., '€'). """
    currency_display_symbol::String

    # --- Capital Constraints ---
    """ Minimum total capital to be invested in the portfolio. """
    minimum_total_investment::Float64
    """ Maximum total capital available for investment. """
    maximum_total_investment::Float64
    """ Minimum transaction size per asset (prevents micro-positions). """
    minimum_trade_size::Float64

    # --- Goals ---
    """ The absolute target profit in real terms (adjusted for inflation) over the horizon. """
    target_real_profit_amount::Float64
    """ Investment horizon in years. """
    investment_horizon_years::Float64

    # --- Economic Assumptions ---
    """ Expected annualized inflation rate (decimal, e.g., 0.02 for 2%). """
    expected_inflation_rate::Float64
    """ The risk-free rate of return (decimal). """
    risk_free_rate::Float64

    # --- Risk Constraints ---
    """ Global maximum weight cap for any single asset (decimal). """
    global_maximum_asset_weight::Float64
    """ Minimum allocation required for safe assets (Bonds/Cash). """
    minimum_safety_allocation_weight::Float64

    # --- Data Settings ---
    """ Number of years of historical data to fetch. """
    historical_data_years::Int
    """ Minimum number of valid trading days required to include a stock. """
    minimum_data_points_required::Int
    """ Intensity for Ledoit-Wolf covariance shrinkage (0.0 to 1.0). """
    covariance_shrinkage_intensity::Float64
    """ Cap on expected annual stock returns to prevent optimizer overfitting. """
    maximum_stock_return_cap::Float64
end

"""
    MarketModel

Container for the processed mathematical model of the market.
"""
struct MarketModel
    """ List of all valid assets included in the model. """
    investable_assets::Vector{AssetDefinition}
    """ Vector of expected real returns (annualized) matching the asset list order. """
    expected_real_returns_vector::Vector{Float64}
    """ Covariance matrix of asset returns. """
    covariance_matrix::Matrix{Float64}
end

"""
    OptimizationResult

Container for the output of the solver.
"""
struct OptimizationResult
    """ The final status code returned by the solver (e.g., OPTIMAL, INFEASIBLE). """
    solver_termination_status::MOI.TerminationStatusCode
    """ Total capital utilized in the solution. """
    total_capital_deployed::Float64
    """ Amount held in risk-free cash. """
    cash_allocation_amount::Float64
    """ Vector of capital allocated to each asset. """
    asset_allocation_amounts::Vector{Float64}
    """ Expected portfolio volatility (standard deviation). """
    portfolio_risk_percentage::Float64
    """ Expected portfolio return (annualized). """
    portfolio_return_percentage::Float64
    """ The Sharpe Ratio of the optimized portfolio. """
    portfolio_sharpe_ratio::Float64
    """ The list of assets corresponding to the allocation vector. """
    assets_reference::Vector{AssetDefinition}
end

# ==============================================================================
#  2. DATA INGESTION LAYER
# ==============================================================================

"""
    parse_optional_numeric_value(value)

Helper to parse CSV fields that might be missing, empty, or 0.
Returns `nothing` if the value implies no constraint.
"""
function parse_optional_numeric_value(input_value)::Union{Float64,Nothing}
    if ismissing(input_value) || input_value === nothing || input_value == ""
        return nothing
    end
    if input_value isa Number
        # Treat 0.0 as "no specific constraint set" in this context
        return input_value == 0.0 ? nothing : Float64(input_value)
    end
    return nothing
end

"""
    load_asset_universe(shares_filepath, bonds_filepath)

Reads CSV files and constructs the universe of `AssetDefinition` objects.
"""
function load_asset_universe(shares_filepath::String, bonds_filepath::String)::Vector{AssetDefinition}
    println("[INFO] Initializing Asset Universe...")

    # Assert files exist before attempting to read
    @assert isfile(shares_filepath) "Shares CSV file not found at: $shares_filepath"
    @assert isfile(bonds_filepath) "Bonds CSV file not found at: $bonds_filepath"

    asset_universe = AssetDefinition[]

    function process_csv_file(filepath::String, category::AssetCategory)
        data_frame = CSV.read(filepath, DataFrame)

        # Validation: Check for required columns based on asset type
        required_columns = (category == Stock) ? ["Ticker", "Name"] : ["Name", "Yield", "Risk"]
        missing_columns = filter(col -> !(col in names(data_frame)), required_columns)

        if !isempty(missing_columns)
            @warn "File $filepath is missing required columns: $missing_columns"
            return
        end

        has_min_col = "Min" in names(data_frame)
        has_max_col = "Max" in names(data_frame)

        for row in eachrow(data_frame)
            user_min = has_min_col ? parse_optional_numeric_value(row.Min) : nothing
            user_max = has_max_col ? parse_optional_numeric_value(row.Max) : nothing

            # Determine ID and properties based on type
            identifier = (category == Stock) ? String(row.Ticker) : String(row.Name)
            is_manual_flag = (category == Bond)
            nominal_yield = (category == Bond) ? Float64(row.Yield) : 0.0
            risk_val = (category == Bond) ? Float64(row.Risk) : 0.0

            push!(asset_universe, AssetDefinition(
                identifier,
                category,
                String(row.Name),
                is_manual_flag,
                nominal_yield,
                risk_val,
                user_min,
                user_max
            ))
        end
        println("   > Loaded $(nrow(data_frame)) $(category)s")
    end

    process_csv_file(shares_filepath, Stock)
    process_csv_file(bonds_filepath, Bond)

    @assert length(asset_universe) > 0 "Asset universe is empty. Check input CSVs."

    return asset_universe
end

# ==============================================================================
#  3. MARKET DATA PROCESSING
# ==============================================================================

"""
    apply_covariance_shrinkage(sample_covariance_matrix, shrinkage_factor)

Applies Ledoit-Wolf style shrinkage to the sample covariance matrix to reduce estimation error.
Formula: (1 - alpha) * Sigma + alpha * Target(Diagonal)
"""
function apply_covariance_shrinkage(sample_covariance_matrix::Matrix{Float64}, shrinkage_factor::Float64)::Matrix{Float64}
    # Validate shrinkage factor is between 0 and 1
    @assert 0.0 <= shrinkage_factor <= 1.0 "Shrinkage factor must be between 0.0 and 1.0"

    target_diagonal_matrix = Diagonal(diag(sample_covariance_matrix))
    shrunk_matrix = (1.0 - shrinkage_factor) * sample_covariance_matrix + shrinkage_factor * target_diagonal_matrix
    return shrunk_matrix
end

"""
    construct_market_model(config, asset_universe)

Orchestrates the data downloading, cleaning, and mathematical modeling (Returns & Covariance).
"""
function construct_market_model(config::PortfolioConfiguration, asset_universe::Vector{AssetDefinition})::MarketModel
    println("[INFO] Building Robust Market Model...")

    # Separate assets into those needing API data (Stocks) and manual entries (Bonds)
    stock_assets = filter(asset -> !asset.is_manual_entry, asset_universe)
    manual_assets = filter(asset -> asset.is_manual_entry, asset_universe)

    # Process stocks to get historical returns and covariance
    stock_nominal_returns, stock_covariance_matrix = download_and_process_stocks(stock_assets, config)

    # Extract yield from manual assets
    manual_nominal_returns = [asset.fixed_nominal_yield for asset in manual_assets]

    # Combine returns vectors
    combined_nominal_returns = vcat(stock_nominal_returns, manual_nominal_returns)

    # Convert to Real Returns (Nominal - Inflation)
    final_real_returns = combined_nominal_returns .- config.expected_inflation_rate

    # Combine covariance matrices
    final_covariance_matrix = assemble_full_covariance_matrix(stock_covariance_matrix, manual_assets, length(stock_assets))

    all_assets_ordered = vcat(stock_assets, manual_assets)

    # Dimension Assertions
    @assert length(all_assets_ordered) == length(final_real_returns)
    @assert size(final_covariance_matrix, 1) == length(all_assets_ordered)

    return MarketModel(all_assets_ordered, final_real_returns, final_covariance_matrix)
end

"""
    download_and_process_stocks(stock_list, config)

Fetches historical prices from Yahoo Finance, computes log returns, and calculates statistics.
"""
function download_and_process_stocks(stock_list::Vector{AssetDefinition}, config::PortfolioConfiguration)
    isempty(stock_list) && return Float64[], Matrix{Float64}(undef, 0, 0)

    ticker_symbols = [s.ticker_symbol for s in stock_list]
    println("   > Fetching data for $(length(ticker_symbols)) stocks...")

    ticker_data_map = Dict{String,DataFrame}()
    progress_bar = Progress(length(ticker_symbols), desc="   > Downloading: ", color=:white)

    for ticker in ticker_symbols
        try
            # Fetch data using YFinance
            raw_data = get_prices(ticker, range="$(config.historical_data_years)y", interval="1d")

            # Validation: Check if 'adjclose' exists and has sufficient length
            if haskey(raw_data, "adjclose") && length(raw_data["adjclose"]) > config.minimum_data_points_required
                dates_vector = Date.(raw_data["timestamp"])
                prices_vector = Float64.(raw_data["adjclose"])

                # Filter out NaNs and Zeros
                is_valid_point = .!isnan.(prices_vector) .& (prices_vector .> 0)

                if sum(is_valid_point) > config.minimum_data_points_required
                    ticker_data_map[ticker] = DataFrame(Date=dates_vector[is_valid_point], Price=prices_vector[is_valid_point])
                end
            end
        catch e
            # Swallow error for individual ticker failures to allow process to continue
        end
        next!(progress_bar)
    end
    finish!(progress_bar)

    valid_tickers = collect(keys(ticker_data_map))
    if isempty(valid_tickers)
        error("[CRITICAL] No valid stock data found. Cannot proceed with optimization.")
    end

    if length(valid_tickers) < length(ticker_symbols)
        @warn "   > Dropped $(length(ticker_symbols) - length(valid_tickers)) failed tickers."
    end

    # Merge DataFrames on Date (Inner Join implies intersection of dates)
    merged_dataframe = ticker_data_map[valid_tickers[1]]
    rename!(merged_dataframe, :Price => Symbol(valid_tickers[1]))

    for ticker in valid_tickers[2:end]
        temp_df = ticker_data_map[ticker]
        rename!(temp_df, :Price => Symbol(ticker))
        merged_dataframe = innerjoin(merged_dataframe, temp_df, on=:Date)
    end
    sort!(merged_dataframe, :Date)

    # --- Mathematical Processing ---
    # Extract price matrix
    price_matrix = Matrix(merged_dataframe[:, valid_tickers])

    # Calculate Log Returns: ln(P_t / P_t-1)
    log_returns_matrix = diff(log.(price_matrix), dims=1)

    # Annualize Returns (Mean * 252 trading days)
    raw_mean_returns_annualized = vec(mean(log_returns_matrix, dims=1)) .* 252

    # Apply Cap to Returns (Heuristic against overfitting)
    nominal_return_cap = config.maximum_stock_return_cap + config.expected_inflation_rate
    final_expected_returns = min.(raw_mean_returns_annualized, nominal_return_cap)

    # Calculate Covariance and Shrink
    sample_covariance = cov(log_returns_matrix) .* 252
    shrunk_covariance = apply_covariance_shrinkage(sample_covariance, config.covariance_shrinkage_intensity)

    # Sanity Check: Ensure partial failures didn't corrupt the logic
    if length(valid_tickers) != length(stock_list)
        # Note: In a real production system, we might handle this gracefully. 
        # Here we error out to force the user to clean their CSV.
        error("[CRITICAL] Data download partial failure. Resulting matrix size mismatch. Please remove failed tickers from CSV.")
    end

    return final_expected_returns, shrunk_covariance
end

"""
    assemble_full_covariance_matrix(stock_cov, manual_assets, num_stocks)

Constructs the block-diagonal covariance matrix combining calculated stock covariance
and manually specified bond risks (assumed uncorrelated).
"""
function assemble_full_covariance_matrix(stock_covariance::Matrix, manual_assets::Vector{AssetDefinition}, num_stocks::Int)
    total_assets_count = num_stocks + length(manual_assets)
    full_covariance = zeros(total_assets_count, total_assets_count)

    # Fill the stock block (top-left)
    if num_stocks > 0
        full_covariance[1:num_stocks, 1:num_stocks] = stock_covariance
    end

    # Fill the manual assets (diagonal elements only, assuming zero correlation with stocks)
    for i in 1:length(manual_assets)
        index_in_matrix = num_stocks + i
        # Variance = Standard Deviation ^ 2
        full_covariance[index_in_matrix, index_in_matrix] = manual_assets[i].annualized_risk_sigma^2
    end

    return full_covariance
end

# ==============================================================================
#  4. OPTIMIZATION ENGINE
# ==============================================================================

"""
    validate_feasibility_constraints(config, assets)

Performs pre-optimization checks to ensure the constraints aren't mathematically impossible.
"""
function validate_feasibility_constraints(config::PortfolioConfiguration, assets::Vector{AssetDefinition})
    total_forced_allocation = 0.0
    for asset in assets
        if !isnothing(asset.minimum_allocation_weight)
            total_forced_allocation += asset.minimum_allocation_weight
        end
    end

    # Assertion: Forced Minimums + Safety Floor cannot exceed 100%
    if total_forced_allocation + config.minimum_safety_allocation_weight > 1.0 + 1e-9
        error("[CONFIG ERROR] Sum of forced asset minimums + safety allocation exceeds 100%. Optimization is impossible.")
    end
end

"""
    execute_mean_variance_optimization(config, market_model)

Constructs and solves the Mixed-Integer Quadratic Programming (MIQP) problem.
Objective: Minimize Variance
Constraints: Return Target, Budget, Cardinality/Thresholds.
"""
function execute_mean_variance_optimization(config::PortfolioConfiguration,
    data::MarketModel)::OptimizationResult

    println("[INFO] Optimizing Strategy (Correct MIQP)...")

    validate_feasibility_constraints(config, data.investable_assets)

    # Initialize Gurobi Solver
    optimization_model = Model(Gurobi.Optimizer)
    set_silent(optimization_model)
    # 0.1% optimality gap for faster convergence on integer problems
    set_optimizer_attribute(optimization_model, "MIPGap", 0.001)

    num_assets = length(data.investable_assets)

    # --- Decision Variables ---
    # x: Amount of currency invested in each asset
    @variable(optimization_model, asset_investment_vars[1:num_assets] >= 0)

    # x_cash: Amount kept in risk-free liquidity
    @variable(optimization_model, cash_allocation_var >= 0)

    # z: Binary indicator (1 if asset is held, 0 otherwise)
    @variable(optimization_model, asset_selected_indicator[1:num_assets], Bin)

    # K: Total capital to be invested (variable within bounds)
    @variable(optimization_model,
        config.minimum_total_investment <= total_capital_var <= config.maximum_total_investment)

    # --- 1. Budget Constraint ---
    # Sum of all assets + cash must equal total capital deployed
    @constraint(optimization_model, sum(asset_investment_vars) + cash_allocation_var == total_capital_var)

    # --- 2. Objective Function: Minimize Variance ---
    # We minimize x'Qx. Note: We are minimizing total variance dollars squared.
    @objective(optimization_model, Min,
        asset_investment_vars' * data.covariance_matrix * asset_investment_vars)

    # --- 3. Return Constraint ---
    # Real Risk Free Rate = Nominal RF - Inflation
    real_risk_free_rate = config.risk_free_rate - config.expected_inflation_rate

    # Convert absolute profit target to required annual return rate
    target_annual_profit_amount = config.target_real_profit_amount / config.investment_horizon_years

    # Total Return >= Target
    @constraint(optimization_model,
        dot(asset_investment_vars, data.expected_real_returns_vector)
        +
        cash_allocation_var * real_risk_free_rate
        >=
        target_annual_profit_amount)

    # --- 4. Asset Specific Constraints ---
    for i in 1:num_assets
        current_asset = data.investable_assets[i]

        # Determine effective bounds
        effective_max_weight = isnothing(current_asset.maximum_allocation_weight) ?
                               config.global_maximum_asset_weight : current_asset.maximum_allocation_weight

        effective_min_weight = isnothing(current_asset.minimum_allocation_weight) ?
                               0.0 : current_asset.minimum_allocation_weight

        # Hard upper limit based on total capital
        @constraint(optimization_model, asset_investment_vars[i] <= effective_max_weight * total_capital_var)

        if effective_min_weight > 0
            # If a specific minimum is set, we force allocation
            @constraint(optimization_model, asset_investment_vars[i] >= effective_min_weight * total_capital_var)
            @constraint(optimization_model, asset_selected_indicator[i] == 1)
        else
            # Semi-continuous constraint:
            # If z=1, then min_trade <= x <= max_invest
            # If z=0, then x = 0
            @constraint(optimization_model, asset_investment_vars[i] >= config.minimum_trade_size * asset_selected_indicator[i])
            @constraint(optimization_model, asset_investment_vars[i] <= config.maximum_total_investment * asset_selected_indicator[i])
        end
    end

    # --- 5. Safety Floor Constraint ---
    # Ensure bonds + cash >= safety_percentage
    bond_indices = findall(a -> a.category == Bond, data.investable_assets)

    if !isempty(bond_indices)
        @constraint(optimization_model,
            sum(asset_investment_vars[bond_indices]) + cash_allocation_var
            >=
            config.minimum_safety_allocation_weight * total_capital_var)
    else
        # If no bonds exist, cash must satisfy the safety requirement
        @constraint(optimization_model,
            cash_allocation_var >= config.minimum_safety_allocation_weight * total_capital_var)
    end

    # --- Solve ---
    optimize!(optimization_model)

    status = termination_status(optimization_model)

    if status != MOI.OPTIMAL
        return OptimizationResult(status, 0.0, 0.0, Float64[],
            0.0, 0.0, 0.0, AssetDefinition[])
    end

    # --- Extract Results ---
    final_asset_allocations = value.(asset_investment_vars)
    final_cash_allocation = value(cash_allocation_var)
    final_total_capital = value(total_capital_var)

    # Calculate Portfolio Metrics
    variance_dollars = final_asset_allocations' * data.covariance_matrix * final_asset_allocations
    portfolio_volatility_percent = sqrt(max(variance_dollars, 0.0)) / final_total_capital

    portfolio_return_percent =
        (dot(final_asset_allocations, data.expected_real_returns_vector)
         +
         final_cash_allocation * real_risk_free_rate) / final_total_capital

    sharpe_ratio_value =
        portfolio_volatility_percent > 1e-9 ?
        (portfolio_return_percent - real_risk_free_rate) / portfolio_volatility_percent : 0.0

    return OptimizationResult(
        status,
        final_total_capital,
        final_cash_allocation,
        final_asset_allocations,
        portfolio_volatility_percent,
        portfolio_return_percent,
        sharpe_ratio_value,
        data.investable_assets
    )
end

# ==============================================================================
#  5. REPORTING
# ==============================================================================

"""
    generate_text_report(config, result)

Prints a formatted CLI summary of the optimization result.
"""
function generate_text_report(config::PortfolioConfiguration, result::OptimizationResult)
    if result.solver_termination_status != MOI.OPTIMAL
        println("[ERROR] Infeasible. Target profit may be too high for risk constraints.")
        return
    end

    currency_sym = config.currency_display_symbol

    projected_total_profit = (result.portfolio_return_percentage * result.total_capital_deployed) * config.investment_horizon_years

    println("\n" * "="^70)
    println("                  STRATEGY REPORT (LDI)")
    println("="^70)

    @printf("Total Investment:    %s%s\n", currency_sym, string(round(Int, result.total_capital_deployed)))
    @printf("Real Profit Target:  %s%s\n", currency_sym, string(round(Int, config.target_real_profit_amount)))
    @printf("Est. Real Profit:    %s%s\n", currency_sym, string(round(Int, projected_total_profit)))
    println("-"^70)
    @printf("Annual Real Return:  %.2f%%\n", result.portfolio_return_percentage * 100)
    @printf("Annual Risk (Vol):   %.2f%%\n", result.portfolio_risk_percentage * 100)
    @printf("Sharpe Ratio:        %.2f\n", result.portfolio_sharpe_ratio)
    println("-"^70)

    # Build DataFrame for display
    display_dataframe = DataFrame(
        Name=[asset.display_name for asset in result.assets_reference],
        Type=[string(asset.category) for asset in result.assets_reference],
        Value=result.asset_allocation_amounts,
        Weight=result.asset_allocation_amounts ./ result.total_capital_deployed
    )

    # Add Cash row if significant
    if result.cash_allocation_amount > 1.0
        label = "$(config.base_currency_code) Liquidity (Risk Free)"
        push!(display_dataframe, (label, "Cash", result.cash_allocation_amount, result.cash_allocation_amount / result.total_capital_deployed))
    end

    # Filter out trivial allocations (less than 0.1%)
    filter!(row -> row.Weight > 0.001, display_dataframe)
    sort!(display_dataframe, :Weight, rev=true)

    for row in eachrow(display_dataframe)
        @printf("%-10s | %-25s | %6.2f%% | %s%s\n",
            row.Type, row.Name, row.Weight * 100, currency_sym, string(round(Int, row.Value)))
    end
    println("="^70)
end

"""
    generate_chart_visualization(result, config, output_filename)

Generates a donut chart of the portfolio allocation using CairoMakie.
"""
function generate_chart_visualization(result::OptimizationResult, config::PortfolioConfiguration, output_filename::String)
    if result.solver_termination_status != MOI.OPTIMAL
        return
    end
    println("[INFO] Generating Visualization...")

    asset_names = [asset.display_name for asset in result.assets_reference]
    asset_values = copy(result.asset_allocation_amounts)

    # Add Cash slice
    if result.cash_allocation_amount > 1.0
        push!(asset_names, "$(config.base_currency_code) Liquidity")
        push!(asset_values, result.cash_allocation_amount)
    end

    normalized_weights = asset_values ./ sum(asset_values)

    # Filter for visualization clarity (threshold 1%)
    is_visible = normalized_weights .> 0.01

    # Sort largest to smallest
    sort_permutation = sortperm(normalized_weights[is_visible], rev=true)
    final_names = asset_names[is_visible][sort_permutation]
    final_weights = normalized_weights[is_visible][sort_permutation]

    # Plotting
    color_palette = get(ColorSchemes.viridis, range(0, 1, length=length(final_weights)))
    figure = Figure(size=(1000, 600), backgroundcolor=:white)
    axis = Axis(figure[1, 1], aspect=DataAspect())
    hidedecorations!(axis)
    hidespines!(axis)

    pie!(axis, final_weights, color=color_palette, inner_radius=0.5, strokewidth=2, strokecolor=:white)

    legend_elements = [PolyElement(color=c, strokecolor=:transparent) for c in color_palette]
    legend_labels = ["$n ($(round(w*100, digits=1))%)" for (n, w) in zip(final_names, final_weights)]
    Legend(figure[1, 2], legend_elements, legend_labels, "Allocation", framevisible=false)

    save(output_filename, figure)
    println("   > Chart saved to $output_filename")
end

# ==============================================================================
#  6. MAIN EXECUTION
# ==============================================================================

function run_pipeline()
    data_directory = joinpath(@__DIR__, "data")
    shares_csv_path = joinpath(data_directory, "shares.csv")
    bonds_csv_path = joinpath(data_directory, "bonds.csv")

    # 1. Load Data
    asset_universe = load_asset_universe(shares_csv_path, bonds_csv_path)

    # 2. Configuration
    configuration = PortfolioConfiguration(
        # --- Localization ---
        "EUR",                  # Base Currency Code
        "€",                    # Currency Display Symbol

        # --- Capital ---
        80_000.0,               # Minimum Investment
        100_000.0,              # Maximum Investment
        1000.0,                 # Minimum Trade Size

        # --- Goals ---
        20_000.0,               # Target Real Profit Amount
        2.0,                    # Horizon Years

        # --- Economics ---
        0.025,                  # Expected Inflation (2.5%)
        0.000,                  # Risk Free Rate (0.0%)

        # --- Risk ---
        0.25,                   # Global Maximum Asset Weight
        0.10,                   # Minimum Safety Allocation

        # --- Data Settings ---
        3,                      # Historical Data Years
        50,                     # Min Data Points Required
        0.20,                   # Covariance Shrinkage Intensity
        0.12                    # Max Stock Return Cap
    )

    try
        # 3. Processing
        market_model = construct_market_model(configuration, asset_universe)

        # 4. Optimization
        optimization_result = execute_mean_variance_optimization(configuration, market_model)

        # 5. Output
        generate_text_report(configuration, optimization_result)
        generate_chart_visualization(optimization_result, configuration, "portfolio_strategy.png")

    catch error_instance
        println("\n[FATAL ERROR] Pipeline failed.")
        showerror(stdout, error_instance)
    end
end

end # module

Pecunia.run_pipeline()