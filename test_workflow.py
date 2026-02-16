import os
import json
from workflow import workflow, MarketingState, CampaignInput

def run_test(test_name, campaigns, optimization_targets=None):
    print(f"\n{'='*20} Running Test: {test_name} {'='*20}")
    
    state_input = {
        "campaigns": campaigns,
        "iteration": 0,
        "max_iterations": 2
    }
    
    if optimization_targets:
        state_input["optimization_targets"] = optimization_targets
    
    try:
        final_state = workflow.invoke(state_input)
        print(f"Test {test_name} completed successfully.")
        
        # Print some summary info
        print(f"Final Iteration: {final_state.get('iteration')}")
        print(f"Strategy Report length: {len(final_state.get('strategy_report', ''))}")
        
        # Check if recommendations were generated
        recommendations = final_state.get('recommendations')
        if recommendations and isinstance(recommendations, list):
            print(f"Recommendations generated for {len(recommendations)} campaigns.")
            for rec in recommendations:
                print(f"  - {rec.get('campaign_name')}: {rec.get('recommended_budget')}")
        elif recommendations:
            print(f"Recommendations generated: {recommendations}")
        
        return final_state
    except Exception as e:
        print(f"Test {test_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test Case 1: Standard Case (3 campaigns)
    campaigns_v1 = [
        CampaignInput(
            campaign_id="C1",
            campaign_name="Summer Sale FB",
            channel="Facebook",
            budget=5000.0,
            status="active",
            impressions=100000,
            clicks=2000,
            conversions=100,
            spend=4500.0,
            revenue=15000.0,
            bounces=500,
            total_customers_start=1000,
            total_customers_end=1050,
            lost_customers=50,
            avg_customer_lifespan=12.0,
            start_date="2026-06-01",
            end_date="2026-06-30"
        ),
        CampaignInput(
            campaign_id="C2",
            campaign_name="Google Search Prod",
            channel="Google Ads",
            budget=8000.0,
            status="active",
            impressions=50000,
            clicks=1500,
            conversions=80,
            spend=7500.0,
            revenue=20000.0,
            bounces=300,
            total_customers_start=2000,
            total_customers_end=2020,
            lost_customers=60,
            avg_customer_lifespan=18.0,
            start_date="2026-06-01",
            end_date="2026-06-30"
        )
    ]
    
    # Test Case 2: Edge Case (Zero impressions/clicks)
    campaigns_v2 = [
        CampaignInput(
            campaign_id="C3",
            campaign_name="New Launch IG",
            channel="Instagram",
            budget=1000.0,
            status="active",
            impressions=0,
            clicks=0,
            conversions=0,
            spend=0.0,
            revenue=0.0,
            bounces=0,
            total_customers_start=100,
            total_customers_end=100,
            lost_customers=0,
            avg_customer_lifespan=6.0,
            start_date="2026-07-01",
            end_date="2026-07-07"
        )
    ]
    
    # Test Case 3: High Risk / Poor Performance
    campaigns_v3 = [
        CampaignInput(
            campaign_id="C4",
            campaign_name="LinkedIn B2B Poor",
            channel="LinkedIn",
            budget=10000.0,
            status="active",
            impressions=10000,
            clicks=50,
            conversions=1,
            spend=9000.0,
            revenue=500.0,
            bounces=45,
            total_customers_start=500,
            total_customers_end=480,
            lost_customers=30,
            avg_customer_lifespan=24.0,
            start_date="2026-05-01",
            end_date="2026-05-31"
        )
    ]

    # Run tests
    run_test("Standard", campaigns_v1)
    run_test("Edge Case Zero", campaigns_v2)
    run_test("High Risk", campaigns_v3)
    
    # Test Case 4: Validation Error (Clicks > Impressions)
    print("\n" + "="*20 + " Running Test: Validation Error " + "="*20)
    try:
        invalid_campaign = CampaignInput(
            campaign_id="ERR1",
            campaign_name="Invalid Clicks",
            channel="Facebook",
            budget=1000.0,
            status="active",
            impressions=100,
            clicks=200, # Invalid
            conversions=10,
            spend=500.0,
            revenue=1000.0,
            bounces=10,
            total_customers_start=100,
            total_customers_end=100,
            lost_customers=0,
            avg_customer_lifespan=12.0,
            start_date="2026-01-01",
            end_date="2026-01-31"
        )
        print("Error: CampaignInput should have failed validation!")
    except ValueError as e:
        print(f"Success: Caught expected validation error: {e}")
    
    # Test Case 5: Custom Optimization Target
    run_test("Custom Target", campaigns_v1, optimization_targets={"Conversion Rate": 1.0, "Return on Ad Spend": 0.5})
