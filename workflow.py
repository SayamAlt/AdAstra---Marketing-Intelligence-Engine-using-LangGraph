import os
import cvxpy as cp
import numpy as np
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, computed_field, model_validator
from typing import Optional, Literal, List, Dict
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.caches import InMemoryCache
from langgraph.graph import StateGraph, START, END
import streamlit as st

load_dotenv()

if "secrets" in st.secrets:
    OPENAI_API_KEY = st.secrets["secrets"]["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the cache
cache = InMemoryCache()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.6, cache=cache)

# Define Pydantic models
class CampaignInput(BaseModel):
    campaign_id: str = Field(..., description="The unique ID of the campaign")
    campaign_name: str = Field(..., description="The name of the campaign")
    channel: Literal["Facebook", "Google Ads", "Instagram", "LinkedIn", "Twitter", "Email", "Other"] = Field(..., description="The channel of the campaign")
    budget: float = Field(..., ge=0.0, description="The budget for the campaign")
    status: Literal["active", "paused", "completed", "archived"] = Field(..., description="The status of the campaign")
    impressions: int = Field(..., ge=0, description="The total number of impressions for the campaign")
    clicks: int = Field(..., ge=0, description="The total number of clicks for the campaign")
    conversions: int = Field(..., ge=0, description="The total number of conversions for the campaign")
    spend: float = Field(..., ge=0, description="The total spend for the campaign")
    revenue: float = Field(..., ge=0, description="The total revenue for the campaign")
    bounces: int = Field(..., ge=0, description="The total number of bounces for the campaign")
    total_customers_start: int = Field(..., ge=0, description="The total number of customers at the start of the campaign")
    total_customers_end: int = Field(..., ge=0, description="The total number of customers at the end of the campaign")
    lost_customers: int = Field(..., ge=0, description="The total number of lost customers for the campaign")
    avg_customer_lifespan: float = Field(..., ge=0, description="The average customer lifespan for the campaign")
    currency: str = Field(default="USD", description="The currency of the campaign")
    leads: Optional[int] = Field(default=0, ge=0, description="The total number of leads for the campaign")
    start_date: str = Field(..., description="The start date of the campaign")
    end_date: str = Field(..., description="The end date of the campaign")
    utm_source: Optional[str] = Field(default=None, description="The UTM source for the campaign")
    utm_medium: Optional[str] = Field(default=None, description="The UTM medium for the campaign")
    
    @model_validator(mode='after')
    def validate_metrics_consistency(self) -> 'CampaignInput':
        if self.clicks > self.impressions:
            raise ValueError(f"Clicks ({self.clicks}) cannot exceed impressions ({self.impressions})")
        if self.conversions > self.clicks:
            raise ValueError(f"Conversions ({self.conversions}) cannot exceed clicks ({self.clicks})")
        if self.bounces > self.impressions:
            raise ValueError(f"Bounces ({self.bounces}) cannot exceed impressions ({self.impressions})")
        if self.spend > self.budget * 2: # Sanity check for massive overspend
            pass 
        return self
    
    @computed_field(return_type=float)
    @property
    def click_through_rate(self) -> float:
        return (self.clicks / self.impressions) if self.impressions > 0 else 0.0

    @computed_field(return_type=float)
    @property
    def conversion_rate(self) -> float:
        return (self.conversions / self.clicks) if self.clicks > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def cost_per_click(self) -> float:
        return (self.spend / self.clicks) if self.clicks > 0 else 0.0

    @computed_field(return_type=float)
    @property
    def cost_per_mille(self) -> float:
        return (self.spend / self.impressions) * 1000 if self.impressions > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def return_on_ad_spend(self) -> float:
        return (self.revenue / self.spend) if self.spend > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def cost_per_acquisition(self) -> float:
        return (self.spend / self.conversions) if self.conversions > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def cost_per_lead(self) -> float:
        return (self.spend / self.leads) if self.leads and self.leads > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def average_order_value(self) -> float:
        return (self.revenue / self.conversions) if self.conversions > 0 else 0.0

    @computed_field(return_type=float)
    @property
    def profit(self) -> float:
        return self.revenue - self.spend
    
    @computed_field(return_type=float)
    @property
    def return_on_investment(self) -> float:
        return ((self.revenue - self.spend) / self.spend) if self.spend > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def budget_utilization(self) -> float:
        return (self.spend / self.budget) if self.budget > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def customer_lifetime_value(self) -> float:
        aov = (self.revenue / self.conversions) if self.conversions > 0 else 0.0
        return aov * self.avg_customer_lifespan
    
    @computed_field(return_type=float)
    @property
    def churn_rate(self) -> float:
        return (self.lost_customers / self.total_customers_start) if self.total_customers_start > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def retention_rate(self) -> float:
        return (self.total_customers_end / self.total_customers_start) if self.total_customers_start > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def profit_margin(self) -> float:
        return ((self.revenue - self.spend) / self.revenue) if self.revenue > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def bounce_rate(self) -> float:
        return (self.bounces / self.impressions) if self.impressions > 0 else 0.0
    
    @computed_field(return_type=float)
    @property
    def cost_per_bounce(self) -> float:
        return (self.spend / self.bounces) if self.bounces > 0 else 0.0
    
class IndustryBenchmarks(BaseModel):
    click_through_rate: float = Field(..., description="The click-through rate for the industry")
    conversion_rate: float = Field(..., description="The conversion rate for the industry")
    cost_per_click: float = Field(..., description="The cost per click for the industry")
    cost_per_acquisition: float = Field(..., description="The cost per acquisition for the industry")
    return_on_ad_spend: float = Field(..., description="The return on ad spend for the industry")
    bounce_rate: float = Field(..., description="The bounce rate for the industry")
    profit_margin: float = Field(..., description="The profit margin for the industry")
    
class KPIAnalysis(BaseModel):
    metric: str = Field(..., description="The name of the KPI metric")
    value: float = Field(..., description="The value of the KPI metric")
    benchmark: float = Field(..., description="The benchmark value of the KPI metric")
    rating: Literal["Low", "Medium", "High"] = Field(..., description="KPI metric's performance rating against the benchmark value")
    recommendation: str = Field(..., description="Actionable recommendations to improve or maintain the KPI metric")
    
class CreativeChannelMix(BaseModel):
    suggestions: Optional[List[str]] = Field(default_factory=list, description="Actionable suggestions for creative assets or channel mix")
    
class CampaignDiagnostics(BaseModel):
    kpi_analysis: List[KPIAnalysis] = Field(..., description="Detailed KPI analysis for each campaign")
    overall_health: Literal["Healthy", "Needs Attention", "Underperforming"] = Field(..., description="Overall campaign health based on all KPI metrics")
    
class CampaignRecommendation(BaseModel):
    campaign_id: str = Field(..., description="The ID of the campaign")
    campaign_name: str = Field(..., description="The name of the campaign")
    current_budget: float = Field(..., description="The current budget of the campaign")
    recommended_budget: float = Field(..., description="The recommended budget for the campaign")
    reasoning: str = Field(..., description="The reasoning behind the recommendation")
    metric_values: Optional[Dict[str, float]] = Field(default_factory=dict, description="The recommended metric values for the campaign")
    metric_weights: Optional[Dict[str, float]] = Field(default_factory=dict, description="The weights assigned with each metric for the campaign")

class CampaignRecommendations(BaseModel):
    recommendations: List[CampaignRecommendation] = Field(..., description="A list of budget recommendations for each campaign")
    
class MarketingState(BaseModel):
    campaigns: List[CampaignInput] = Field(..., description="The marketing campaigns executed by the company")
    benchmarks: Optional[List[Dict]] = Field(default_factory=list, description="The industry benchmarks for each campaign")
    campaign_risks: Optional[List[Dict]] = Field(default_factory=list, description="The risk analysis for each campaign")
    metric_sensitivities: Optional[List[Dict]] = Field(default_factory=list, description="The metric sensitivities for each campaign")
    creative_channel_suggestions: Optional[List[Dict]] = Field(default_factory=list, description="The creative channel suggestions for each campaign")
    insights: Optional[List[Dict]] = Field(default_factory=list, description="The insights obtained by analyzing marketing campaigns")
    recommendations: Optional[List[Dict]] = Field(default_factory=list, description="The recommendations for optimization of marketing campaigns")
    strategy_report: Optional[str] = Field(default=None, description="The strategy report for the marketing campaigns")
    iteration: Optional[int] = Field(default=0, description="The iteration of the optimization process")
    max_iterations: Optional[int] = Field(default=3, description="The maximum number of iterations for the optimization process")
    optimization_targets: Optional[Dict[Literal[
        "Return on Ad Spend", 
        "Return on Investment", 
        "Profit Margin", 
        "Conversion Rate", 
        "Click Through Rate", 
        "Cost per Click", 
        "Cost per Acquisition",
        "Bounce Rate",
        "Average Order Value",
        "Cost per Mille",
        "Customer Lifetime Value",
        "Budget Utilization",
        "Churn Rate",
        "Retention Rate"], float]] = Field(default_factory=lambda: {"Return on Ad Spend": 1.0}, description=(
            "Dictionary specifying metrics to optimize and their weights. "
            "Supports single or multiple metrics. For a single metric, set weight=1.0."
        )
    )
    
llm_with_benchmarks = llm.with_structured_output(IndustryBenchmarks)
llm_with_creative_channel_mix = llm.with_structured_output(CreativeChannelMix)
llm_with_campaign_diagnostics = llm.with_structured_output(CampaignDiagnostics)
llm_with_campaign_recommendations = llm.with_structured_output(CampaignRecommendations)

def synchronize_campaign_inputs(state: MarketingState) -> List[CampaignInput]:
    """  
        Synchronizes and validates user-provided campaign inputs.
        Ensures budgets, metrics, and dates are consistent and non-negatives for further processing.
        Returns a list of validated campaign inputs.
    """
    for campaign in state.campaigns:
        # Ensure non-negative numeric fields
        campaign.budget = max(campaign.budget, 0.0)
        campaign.spend = max(campaign.spend, 0.0)
        campaign.revenue = max(campaign.revenue, 0.0)
        campaign.impressions = max(campaign.impressions, 0)
        campaign.clicks = max(campaign.clicks, 0)
        campaign.conversions = max(campaign.conversions, 0)
        campaign.bounces = max(campaign.bounces, 0)
        campaign.total_customers_start = max(campaign.total_customers_start, 0)
        campaign.total_customers_end = max(campaign.total_customers_end, 0)
        campaign.lost_customers = max(campaign.lost_customers, 0)
        campaign.avg_customer_lifespan = max(campaign.avg_customer_lifespan, 0)
        campaign.leads = max(campaign.leads, 0)
        
        # Normalize currency to upper-case
        campaign.currency = (campaign.currency or "USD").upper()
        
        # Validate dates format
        for attr in ["start_date", "end_date"]:
            if not getattr(campaign, attr):
                setattr(campaign, attr, "2026-01-01") # fallback default date
            
    return {"campaigns": state.campaigns}

def fetch_competitor_benchmarks(state: MarketingState) -> List[Dict]:
    """  
        Dynamically generates industry-specific benchmarks for a given channel using an LLM.
        Metrics include CTR, CVR, CPA, ROAS, CPC, Bounce Rate, and Profit Margin.
    """
    benchmarks = []
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a marketing analytics expert. Generate realistic and reasonable "
            "competitor-specific KPI benchmark values for a given channel, considering "
            "the industry and market trends. You will be provided with the names of the "
            "digital marketing campaigns by channel. Focus on realistic industry ranges. "
            "Metrics to include: CTR, CVR, CPA, ROAS, CPC, Bounce Rate, Profit Margin."
        ),
        HumanMessagePromptTemplate.from_template(
            "Generate KPI benchmarks for the '{channel}' channel in JSON format with keys: "
            "'CTR', 'CVR', 'CPA', 'ROAS', 'CPC', 'BounceRate', 'ProfitMargin'. "
            "Use realistic decimal/floating point numbers, e.g., 0.015 for CTR, 1.2 for CPC, etc."
        )
    ])
    
    for campaign in state.campaigns:
        response = llm_with_benchmarks.invoke(prompt.format_messages(channel=campaign.channel))
        
        try:
            benchmarks_data = response.model_dump()
        except Exception as e:
            benchmarks_data = {}
            
        benchmarks.append(benchmarks_data)
        
    return {"benchmarks": benchmarks}

def analyze_campaign_performance(state: MarketingState):
    insights = []
    benchmarks = state.benchmarks
    
    for i, campaign in enumerate(state.campaigns):
        channel = campaign.channel
        benchmark = benchmarks[i]
        
        campaign_kpis = {
            "CTR": campaign.click_through_rate,
            "CVR": campaign.conversion_rate,
            "CPA": campaign.cost_per_acquisition,
            "ROAS": campaign.return_on_ad_spend,
            "CPC": campaign.cost_per_click,
            "BounceRate": campaign.bounce_rate,
            "ProfitMargin": campaign.profit_margin
        }
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a marketing analytics expert. "
                "Evaluate a digital campaign's KPIs against industry benchmarks "
                "and provide a concise, professional performance analysis and action plan."
            ),
            HumanMessagePromptTemplate.from_template(
                """
                Campaign Info:
                ID: {campaign_id}
                Name: {campaign_name}
                Channel: {channel}
                Budget: {budget} {currency}
                Status: {status}
                
                Performance Metrics:
                {campaign_kpis}
                
                Industry Benchmark Metrics:
                {benchmark}
                
                Instructions:
                - Identify underperforming or overperforming metrics.
                - Generate realistic, actionable recommendations to improve campaign ROI.
                - Suggest adjustments in creative, targeting, budget allocation, bidding, or timing.
                - Be concise and specific. Return output strictly in JSON format:
                {{
                    "kpi_analysis": [
                        {{"metric": "<metric_name>", "value": <value>, "benchmark": <benchmark>, "rating": "<Low/Medium/High>", "recommendation": "<action>" }},
                        ...
                    ],
                    "overall_health": "<Healthy/Needs Attention/Underperforming>"
                }}
                """
            )
        ])
        
        response = llm_with_campaign_diagnostics.invoke(
            prompt.format(
                campaign_id = campaign.campaign_id,
                campaign_name = campaign.campaign_name,
                channel = channel,
                budget = campaign.budget,
                currency = campaign.currency,
                status = campaign.status,
                campaign_kpis = campaign_kpis,
                benchmark = benchmark
            )
        )
        
        campaign_diagnostics = response.model_dump()
        insights.append({
            "campaign_id": campaign.campaign_id,
            "campaign_name": campaign.campaign_name,
            "channel": campaign.channel,
            "analysis": campaign_diagnostics
        })
    
    return {"insights": insights}

def evaluate_campaign_risks(state: MarketingState):
    """  
        Compute a risk score for each campaign based on KPI thresholds.
        Risk is based on underperforming KPIs or high cost metrics.
    """
    risks = []
    
    for campaign in state.campaigns:
        risk_score = 0
        risk_reasons = []
        
        if campaign.click_through_rate < 0.01:
            risk_score += 2
            risk_reasons.append("CTR very low (<1%)")
        if campaign.conversion_rate < 0.02:
            risk_score += 2
            risk_reasons.append("CVR very low (<2%)")
        if campaign.bounce_rate > 0.5:
            risk_score += 1
            risk_reasons.append("Bounce Rate high (>50%)")
        if campaign.cost_per_acquisition > 50:
            risk_score += 1
            risk_reasons.append("CPA very high (>50)")
            
        # Risk classification
        if risk_score >= 5:
            risk_level = "High"
        elif risk_score >= 2:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        risks.append({
            "campaign_id": campaign.campaign_id,
            "campaign_name": campaign.campaign_name,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "reasons": risk_reasons
        })
        
    return {"campaign_risks": risks}

def analyze_metric_sensitivity(state: MarketingState):
    """  
        Determine which metrics most affect ROI/ROAS per campaign.
        Sensitivity is computed as normalized metric / weighted contribution.
    """
    sensitivity_results = []
    selected_metrics = list(state.optimization_targets.keys())
    weights = np.array([state.optimization_targets[metric] for metric in selected_metrics])
    
    for campaign in state.campaigns:
        metric_values = np.array([getattr(campaign, metric.lower().strip().replace(" ", "_"), 0.0) for metric in selected_metrics])
        
        # Sensitivity = (normalized metric * weight) / sum(weighted metrics)
        weighted_metric_sum = np.dot(metric_values, weights)
        
        sensitivities = {}
        
        for i, metric in enumerate(selected_metrics):
            val = float(metric_values[i]) if metric_values[i] is not None else 0.0
            sensitivity = ((val * weights[i]) / weighted_metric_sum) if weighted_metric_sum != 0 else 0.0
            sensitivities[metric] = round(float(sensitivity), 4)
            
        sensitivity_results.append({
            "campaign_id": campaign.campaign_id,
            "campaign_name": campaign.campaign_name,
            "metric_sensitivities": sensitivities
        })
    
    return {"metric_sensitivities": sensitivity_results}

def analyze_creative_channels(state: MarketingState):
    """   
        Suggest improvements in creative assets or channel mix based on KPI performance.
    """
    suggestions = []
    
    for campaign in state.campaigns:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a marketing consultant. Recommend specific creative and channel improvements "
                "for a campaign based solely on its KPIs."
            ),
            HumanMessagePromptTemplate.from_template(
                """
                Campaign Info:
                Name: {campaign_name}
                Channel: {channel}
                CTR: {ctr}
                CVR: {cvr}
                Bounce Rate: {bounce_rate}
                ROAS: {roas}

                Provide 2-3 actionable suggestions for improving performance.
                Return output as JSON with key 'suggestions' as a list of strings.
                """
            )
        ])
        
        # Invoke LLM
        response = llm_with_creative_channel_mix.invoke(
            prompt.format(
                campaign_name=campaign.campaign_name,
                channel=campaign.channel,
                ctr=campaign.click_through_rate,
                cvr=campaign.conversion_rate,
                bounce_rate=campaign.bounce_rate,
                roas=campaign.return_on_ad_spend
            )
        )
        
        try:
            suggestions_data = response.model_dump()
        except:
            suggestions_data = {"suggestions": ["No suggestions generated."]}
        
        suggestions.append({
            "campaign_id": campaign.campaign_id,
            "campaign_name": campaign.campaign_name,
            "creative_channel_suggestions": suggestions_data.get("suggestions", [])
        })
        
    return {"creative_channel_suggestions": suggestions}

def optimize_campaign_budget(state: MarketingState):
    """  
        Optimize campaign budgets based on the selected optimization KPI metric. 
    """
    num_campaigns = len(state.campaigns)
    spend_variable = cp.Variable(num_campaigns)
    
    # Extract metrics and weights
    selected_metrics = list(state.optimization_targets.keys())
    weights = np.array([state.optimization_targets[metric] for metric in selected_metrics])
    
    # Build campaign X metric matrix
    metrics_matrix = np.zeros((num_campaigns, len(selected_metrics)))
    
    for i, metric in enumerate(selected_metrics):
        metric_name = metric.lower().strip().replace(" ", "_")
        
        for j, campaign in enumerate(state.campaigns):
            metrics_matrix[j, i] = getattr(campaign, metric_name, 0.0)
            
    # Weighted sum of metrics per campaign
    weighted_metrics = metrics_matrix @ weights
    
    # Define optimization problem
    minimize_metrics = {"cost_per_click", "cost_per_acquisition", "bounce_rate", "cost_per_mille", "churn_rate"}
    
    if all(metric.lower().replace(" ", "_") in minimize_metrics for metric in selected_metrics):
        objective = cp.Minimize(weighted_metrics @ spend_variable)
    else:
        objective = cp.Maximize(weighted_metrics @ spend_variable)
        
    # Enforce constraints
    total_budget = sum(campaign.budget for campaign in state.campaigns)
    constraints = [
        cp.sum(spend_variable) == total_budget, # Total budget constraint
        spend_variable >= 0, # Non-negative budget constraint
    ]
    
    # Prevent large deviations (+/-50%) or handle zero spend
    for i, campaign in enumerate(state.campaigns):
        ref_value = campaign.budget if campaign.spend == 0 else campaign.spend
        constraints.append(spend_variable[i] <= 2.0 * ref_value)
        constraints.append(spend_variable[i] >= 0.1 * ref_value)
        
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
             # Fallback: maintain current budget distribution
             recommended_budgets = [float(campaign.budget) for campaign in state.campaigns]
        else:
             recommended_budgets = [float(spend_variable.value[i]) for i in range(num_campaigns)]
    except Exception as e:
        print(f"Optimization failed: {e}. Falling back to original budgets.")
        recommended_budgets = [float(campaign.budget) for campaign in state.campaigns]
    
    # Generate recommendations
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            You are a marketing campaign optimization assistant.
            Your task is to generate actionable budget recommendations per campaign.
            Consider the metrics and weights provided.
            Provide a concise reasoning for each recommended budget.
            Output strictly in JSON matching the CampaignRecommendation schema.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Campaigns Data:
            {campaigns_data}

            Recommended Budgets (from optimization):
            {recommended_budgets}

            Metrics used: {metrics}
            Metric weights: {weights}

            Generate structured recommendations with reasoning for each campaign.
            """
        )
    ])
    
    campaigns_data = [
        {
            "campaign_id": campaign.campaign_id,
            "campaign_name": campaign.campaign_name,
            "current_budget": campaign.budget,
            "metrics": {metric: float(getattr(campaign, metric.lower().strip().replace(" ", "_"), 0.0)) for metric in selected_metrics}
        }
        for campaign in state.campaigns
    ]
    

    
    # Invoke LLM
    response = llm_with_campaign_recommendations.invoke(
        prompt.format(
            campaigns_data=campaigns_data,
            recommended_budgets=recommended_budgets,
            metrics=selected_metrics,
            weights=weights.tolist()
        )
    )
    
    campaign_recommendations = [rec.model_dump() for rec in response.recommendations]
    state.iteration += 1 # Increment iteration
    return {"recommendations": campaign_recommendations, "iteration": state.iteration}

def generate_strategy(state: MarketingState):
    """   
        Generate a strategy report based on the insights and recommendations.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a Senior Marketing Strategist specializing in AI-driven campaign optimization."
            "Your task is to generate a strategy report based on the insights and recommendations provided."
        ),
        HumanMessagePromptTemplate.from_template(
            """  
            Industry Benchmarks:
            {benchmarks}
            
            Campaign Risks:
            {risks}
            
            Campaign Creative and Channel Suggestions:
            {creative_channel_suggestions}
            
            Campaign Metric Sensitivities:
            {metric_sensitivities}
            
            Campaign Diagnostics:
            {diagnostics}

            Optimization Recommendations:
            {recommendations}

            Generate a concise strategy report including:
            - Key Insights
            - Budget Reallocation Justification
            - Suggested Creative and Channel Improvements
            """
        )
    ])
    
    strategy_report = llm.invoke(
        prompt.format(
            benchmarks=state.benchmarks,
            risks=state.campaign_risks,
            creative_channel_suggestions=state.creative_channel_suggestions,
            metric_sensitivities=state.metric_sensitivities,
            diagnostics=state.insights,
            recommendations=state.recommendations
        )
    ).content
    return {"strategy_report": strategy_report}

def check_marketing_performance(state: MarketingState):
    """  
        Determine whether to continue optimizing or finalize strategy.
        Stop if ROAS >= 3 or max_iterations reached. 
    """
    avg_roas = np.mean([campaign.return_on_ad_spend for campaign in state.campaigns])

    if avg_roas >= 3 or state.iteration >= state.max_iterations:
        return "finalize"
    
    return "optimize_again"

# Define a state graph
graph = StateGraph(MarketingState)

# Add nodes to the graph
graph.add_node("synchronize_campaign_inputs", synchronize_campaign_inputs)
graph.add_node("analyze_campaign_performance", analyze_campaign_performance)
graph.add_node("optimize_campaign_budget", optimize_campaign_budget)
graph.add_node("generate_strategy", generate_strategy)
graph.add_node("fetch_competitor_benchmarks", fetch_competitor_benchmarks)
graph.add_node("evaluate_campaign_risks", evaluate_campaign_risks)
graph.add_node("analyze_metric_sensitivity", analyze_metric_sensitivity)
graph.add_node("analyze_creative_channels", analyze_creative_channels)

# Add edges to the graph
graph.add_edge(START, "synchronize_campaign_inputs")
graph.add_edge("synchronize_campaign_inputs", "fetch_competitor_benchmarks")

# Define parallel edges
graph.add_edge("fetch_competitor_benchmarks", "analyze_campaign_performance")
graph.add_edge("fetch_competitor_benchmarks", "evaluate_campaign_risks")
graph.add_edge("fetch_competitor_benchmarks", "analyze_metric_sensitivity")
graph.add_edge("fetch_competitor_benchmarks", "analyze_creative_channels")

# Merge parallel analysis nodes into budget optimization
graph.add_edge("analyze_campaign_performance", "optimize_campaign_budget")
graph.add_edge("evaluate_campaign_risks", "optimize_campaign_budget")
graph.add_edge("analyze_metric_sensitivity", "optimize_campaign_budget")
graph.add_edge("analyze_creative_channels", "optimize_campaign_budget")

# Add sequential edge
graph.add_edge("optimize_campaign_budget", "generate_strategy")

# Add conditional edges for optimization
graph.add_conditional_edges(
    "generate_strategy",
    check_marketing_performance,
    {
        "finalize": END,
        "optimize_again": "optimize_campaign_budget"
    }
)

# Compile the graph
workflow = graph.compile()

# Visualize the workflow
try:
    with open("marketing_campaign_optimization_workflow.png", "wb") as f:
        f.write(workflow.get_graph().draw_mermaid_png())
    print("Workflow visualization saved as 'marketing_campaign_optimization_workflow.png'")
except Exception as e:
    print(f"Could not save workflow visualization: {e}")