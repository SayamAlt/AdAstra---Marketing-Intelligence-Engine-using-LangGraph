import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from workflow import workflow, CampaignInput, MarketingState
import json, sys, io, re, tempfile, os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Set page configuration
st.set_page_config(
    page_title="AdAstra | Hyper-Optimized Marketing Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Persistence configuration
PERSISTENCE_FILE = "dashboard_state.json"

def load_persisted_state():
    if os.path.exists(PERSISTENCE_FILE):
        try:
            with open(PERSISTENCE_FILE, "r") as f:
                data = json.load(f)
                campaigns_raw = data.get("campaigns", [])
                campaigns = [CampaignInput(**c) for c in campaigns_raw]
                results = data.get("results")
                
                if results and 'campaigns' in results:
                    results['campaigns'] = [CampaignInput(**c) for c in results['campaigns']]
                    
                return campaigns, results
        except Exception as e:
            print(f"Error loading persisted data: {e}")
            if os.path.exists(PERSISTENCE_FILE):
                try: os.remove(PERSISTENCE_FILE)
                except: pass
    return [], None

def save_state_to_disk(campaigns, results=None):
    try:
        # Convert Pydantic models to dicts if needed
        serializable_results = results
        if hasattr(results, "model_dump"):
            serializable_results = results.model_dump()
        
        with open(PERSISTENCE_FILE, "w") as f:
            json.dump({"campaigns": campaigns, "results": serializable_results}, f)
    except Exception as e:
        print(f"Error saving data: {e}")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 50%, #fce7f3 100%);
        background-attachment: fixed;
        background-size: 200% 200%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        padding: 2.5rem 2rem;
        border-radius: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.15);
        margin-bottom: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        text-align: center;
        animation: slideDown 0.8s ease-out;
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    .metric-card h2 {
        color: #1e293b;
        font-weight: 800;
        margin: 0.5rem 0 0 0;
    }
    
    .metric-card small {
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        border-radius: 1rem;
        font-weight: 700;
        transition: all 0.3s ease;
        text-transform: uppercase;
        font-size: 0.9rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover:before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    .form-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f9ff 0%, #e0e7ff 100%);
        backdrop-filter: blur(20px);
        border-right: 2px solid #667eea;
        box-shadow: 4px 0 20px rgba(102, 126, 234, 0.1);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown {
        color: #1e293b !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #334155 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
        border-bottom: 3px solid #667eea;
        padding: 1rem;
        border-radius: 1rem 1rem 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 700;
        color: #1e293b;
        background: rgba(255, 255, 255, 0.8);
        padding: 0.75rem 1.5rem;
        border-radius: 0.75rem;
        transition: all 0.3s;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        color: white !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-color: #667eea !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .pulse-animation {
        animation: pulse 2s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# PDF Generation with ReportLab
def create_pdf_report(title, content):
    """Generate PDF report using ReportLab with markdown parsing"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=12,
        spaceBefore=16,
        fontName='Helvetica-Bold'
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=16,
        spaceAfter=8,
        textColor=colors.HexColor('#334155')
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['BodyText'],
        fontSize=11,
        leading=16,
        spaceAfter=6,
        leftIndent=20,
        textColor=colors.HexColor('#334155')
    )
    
    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Parse markdown content
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Escape special characters
        line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Parse markdown headers
        if line.startswith('# ') and not line.startswith('##'):
            text = line[2:].strip()
            story.append(Paragraph(text, h1_style))
        elif line.startswith('## '):
            text = line[3:].strip()
            story.append(Paragraph(text, h2_style))
        # Parse bullet lists
        elif line.startswith('- ') or line.startswith('* '):
            text = '• ' + line[2:].strip()
            # Handle bold text in bullets
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            story.append(Paragraph(text, bullet_style))
        # Regular paragraphs
        else:
            # Handle bold text
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
            # Handle italic text
            text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
            story.append(Paragraph(text, body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_analytics_pdf(results):
    """Generate Analytics Dashboard PDF with charts using ReportLab"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, 
                                 textColor=colors.HexColor('#667eea'), spaceAfter=20, alignment=TA_CENTER)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, 
                                   textColor=colors.HexColor('#764ba2'), spaceAfter=10, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=10, leading=14)
    
    story = []
    story.append(Paragraph("AdAstra Marketing Analytics Dashboard", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Global Metrics
    total_spend = sum(c.spend for c in results['campaigns'])
    total_revenue = sum(c.revenue for c in results['campaigns'])
    total_conversions = sum(c.conversions for c in results['campaigns'])
    total_impressions = sum(c.impressions for c in results['campaigns'])
    total_clicks = sum(c.clicks for c in results['campaigns'])
    avg_roas = total_revenue / total_spend if total_spend > 0 else 0
    
    story.append(Paragraph("Global Performance Overview", heading_style))
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Spend', f'${total_spend:,.2f}'],
        ['Total Revenue', f'${total_revenue:,.2f}'],
        ['Average ROAS', f'{avg_roas:.2f}x'],
        ['Total Conversions', f'{total_conversions:,}'],
        ['Total Impressions', f'{total_impressions:,}'],
        ['Total Clicks', f'{total_clicks:,}']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Track temp files for cleanup after PDF build
    temp_files_to_cleanup = []
    
    # Generate and embed conversion funnel chart
    try:
        
        funnel_data = {
            'Stage': ['Impressions', 'Clicks', 'Conversions'],
            'Count': [total_impressions, total_clicks, total_conversions]
        }
        
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_data['Stage'],
            x=funnel_data['Count'],
            textinfo="value+percent initial",
            marker=dict(color=['#667eea', '#764ba2', '#f093fb'])
        ))
        fig_funnel.update_layout(title="Conversion Funnel", template="plotly_white", height=400, width=600)
        
        try:
            img_bytes = pio.to_image(fig_funnel, format='png', width=600, height=400)
        except Exception as export_error:
            img_bytes = fig_funnel.to_image(format='png', width=600, height=400)
        
        # Save to temp file
        temp_funnel = tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb')
        temp_funnel.write(img_bytes)
        temp_funnel.close()
                
        if os.path.exists(temp_funnel.name) and os.path.getsize(temp_funnel.name) > 0:
            story.append(Paragraph("Conversion Funnel Analysis", heading_style))
            story.append(RLImage(temp_funnel.name, width=5*inch, height=3.3*inch))
            story.append(Spacer(1, 0.2*inch))
            # Add to cleanup list
            temp_files_to_cleanup.append(temp_funnel.name)
        else:
            raise Exception(f"Temp file invalid: exists={os.path.exists(temp_funnel.name)}, size={os.path.getsize(temp_funnel.name) if os.path.exists(temp_funnel.name) else 0}")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[PDF ERROR] Funnel chart failed: {str(e)}", file=sys.stderr)
        print(f"[PDF ERROR] Full traceback:\n{error_details}", file=sys.stderr)
        story.append(Paragraph(f"Conversion Funnel chart error: {str(e)}", body_style))
    
    # Campaign Performance Chart
    try:
        import plotly.express as px
        import plotly.io as pio
        import sys
        
        chart_df_data = [{
            'Name': c.campaign_name, 
            'Spend': c.spend, 
            'Revenue': c.revenue, 
            'ROAS': c.revenue/c.spend if c.spend > 0 else 0
        } for c in results['campaigns']]
        
        fig_scatter = px.scatter(
            pd.DataFrame(chart_df_data), 
            x="Spend", 
            y="Revenue", 
            size="ROAS", 
            color="ROAS",
            hover_name="Name", 
            template="plotly_white",
            color_continuous_scale='Purples',
            title="Campaign Performance: Spend vs Revenue"
        )
        fig_scatter.update_layout(height=400, width=600)
        
        # Try to export image
        try:
            img_bytes = pio.to_image(fig_scatter, format='png', width=600, height=400)
        except Exception as export_error:
            img_bytes = fig_scatter.to_image(format='png', width=600, height=400)
        
        # Save to temp file
        temp_scatter = tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb')
        temp_scatter.write(img_bytes)
        temp_scatter.close()
        
        # Verify file exists and has content
        if os.path.exists(temp_scatter.name) and os.path.getsize(temp_scatter.name) > 0:
            story.append(Paragraph("Campaign Performance Analysis", heading_style))
            story.append(RLImage(temp_scatter.name, width=5*inch, height=3.3*inch))
            story.append(Spacer(1, 0.2*inch))
            # Add to cleanup list
            temp_files_to_cleanup.append(temp_scatter.name)
        else:
            raise Exception(f"Temp file invalid: exists={os.path.exists(temp_scatter.name)}, size={os.path.getsize(temp_scatter.name) if os.path.exists(temp_scatter.name) else 0}")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[PDF ERROR] Scatter chart failed: {str(e)}", file=sys.stderr)
        print(f"[PDF ERROR] Full traceback:\n{error_details}", file=sys.stderr)
        story.append(Paragraph(f"Performance chart error: {str(e)}", body_style))
    
    # Efficiency Frontier: Spend vs Revenue
    try:
        import plotly.express as px
        import plotly.io as pio
        import sys
        
        chart_df_data = [{'Name': c.campaign_name, 'Spend': c.spend, 'Revenue': c.revenue, 'ROAS': c.revenue/c.spend if c.spend > 0 else 0} for c in results['campaigns']]
        
        fig_efficiency = px.scatter(pd.DataFrame(chart_df_data), x="Spend", y="Revenue", size="ROAS", color="ROAS",
                                   hover_name="Name", template="plotly_white", color_continuous_scale='Purples',
                                   title="Efficiency Frontier: Spend vs Revenue")
        fig_efficiency.update_layout(height=400, width=600)
        
        img_bytes = pio.to_image(fig_efficiency, format='png', width=600, height=400)
        temp_efficiency = tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb')
        temp_efficiency.write(img_bytes)
        temp_efficiency.close()
        
        if os.path.exists(temp_efficiency.name) and os.path.getsize(temp_efficiency.name) > 0:
            story.append(PageBreak())
            story.append(Paragraph("Efficiency Frontier Analysis", heading_style))
            story.append(RLImage(temp_efficiency.name, width=5*inch, height=3.3*inch))
            story.append(Spacer(1, 0.2*inch))
            temp_files_to_cleanup.append(temp_efficiency.name)
    except Exception as e:
        pass
    
    # Reach & Impact Bubble Chart
    try:
        bubble_df_data = [{'Name': c.campaign_name, 'Impressions': c.impressions, 'Conversions': c.conversions, 'Spend': c.spend} for c in results['campaigns']]
        
        fig_bubble = px.scatter(pd.DataFrame(bubble_df_data), x="Impressions", y="Conversions", size="Spend", color="Name",
                               template="plotly_white", title="Reach & Impact: Impressions vs Conversions")
        fig_bubble.update_layout(height=400, width=600, showlegend=False)
        
        img_bytes = pio.to_image(fig_bubble, format='png', width=600, height=400)
        temp_bubble = tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb')
        temp_bubble.write(img_bytes)
        temp_bubble.close()
        
        if os.path.exists(temp_bubble.name) and os.path.getsize(temp_bubble.name) > 0:
            story.append(Paragraph("Reach & Impact Analysis", heading_style))
            story.append(RLImage(temp_bubble.name, width=5*inch, height=3.3*inch))
            story.append(Spacer(1, 0.2*inch))
            temp_files_to_cleanup.append(temp_bubble.name)
    except Exception as e:
        pass
    
    # Performance Quadrant Matrix
    try:
        quad_df_data = [{'Campaign': c.campaign_name, 'ROAS': c.revenue/c.spend if c.spend > 0 else 0, 'Spend': c.spend, 'Channel': c.channel} for c in results['campaigns']]
        quad_df = pd.DataFrame(quad_df_data)
        
        fig_quad = px.scatter(quad_df, x="Spend", y="ROAS", color="Channel", size="Spend", hover_name="Campaign", 
                             template="plotly_white", title="Performance Quadrant Matrix")
        
        avg_spend = quad_df['Spend'].mean()
        avg_roas_quad = quad_df['ROAS'].mean()
        fig_quad.add_hline(y=avg_roas_quad, line_dash="dash", line_color="gray", annotation_text="Avg ROAS")
        fig_quad.add_vline(x=avg_spend, line_dash="dash", line_color="gray", annotation_text="Avg Spend")
        fig_quad.update_layout(height=400, width=600)
        
        img_bytes = pio.to_image(fig_quad, format='png', width=600, height=400)
        temp_quad = tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb')
        temp_quad.write(img_bytes)
        temp_quad.close()
        
        if os.path.exists(temp_quad.name) and os.path.getsize(temp_quad.name) > 0:
            story.append(PageBreak())
            story.append(Paragraph("Performance Quadrant Matrix", heading_style))
            story.append(RLImage(temp_quad.name, width=5*inch, height=3.3*inch))
            story.append(Spacer(1, 0.2*inch))
            temp_files_to_cleanup.append(temp_quad.name)
    except Exception as e:
        pass
    
    # Budget Reallocation Chart
    try:
        recs = results.get('recommendations', [])
        if recs:
            rec_df = pd.DataFrame(recs)
            
            fig_budget = go.Figure()
            fig_budget.add_trace(go.Bar(name='Current Budget', x=rec_df['campaign_name'], y=rec_df['current_budget'], marker_color='#cbd5e1'))
            fig_budget.add_trace(go.Bar(name='AI-Optimized Budget', x=rec_df['campaign_name'], y=rec_df['recommended_budget'], marker_color='#667eea'))
            fig_budget.update_layout(barmode='group', template="plotly_white", title="Budget Reallocation Strategy",
                                    height=400, width=600, xaxis_title="Campaign", yaxis_title="Budget ($)")
            
            img_bytes = pio.to_image(fig_budget, format='png', width=600, height=400)
            temp_budget = tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb')
            temp_budget.write(img_bytes)
            temp_budget.close()
            
            if os.path.exists(temp_budget.name) and os.path.getsize(temp_budget.name) > 0:
                story.append(Paragraph("Budget Reallocation Strategy", heading_style))
                story.append(RLImage(temp_budget.name, width=5*inch, height=3.3*inch))
                story.append(Spacer(1, 0.2*inch))
                temp_files_to_cleanup.append(temp_budget.name)
    except Exception as e:
        pass
    
    # Campaign Diagnostics
    story.append(PageBreak())
    story.append(Paragraph("Campaign Diagnostics", heading_style))
    
    for ins in results['insights']:
        campaign_text = f"<b>{ins['campaign_name']}</b> ({ins['channel']}) - Status: {ins['analysis']['overall_health']}"
        story.append(Paragraph(campaign_text, body_style))
        story.append(Spacer(1, 0.1*inch))
    
    # Build PDF and cleanup temp files
    try:
        doc.build(story)
    finally:
        # Clean up temp image files
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as cleanup_error:
                pass
    
    buffer.seek(0)
    return buffer

# Initialize Session State
if 'campaigns' not in st.session_state or 'results' not in st.session_state:
    saved_campaigns, saved_results = load_persisted_state()
    if 'campaigns' not in st.session_state:
        st.session_state.campaigns = saved_campaigns
    if 'results' not in st.session_state:
        st.session_state.results = saved_results

# Sidebar
with st.sidebar:
    st.image("ai-marketing-campaign-optimization-logo.png", use_container_width=True)
    st.markdown("<h2 style='text-align: center; color: #1e293b; font-weight: 800; margin: 0.5rem 0;'>AdAstra Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-weight: 600;'>AI-Powered Marketing Optimization</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("<h3 style='color: #1e293b; font-weight: 700;'>🎯 Target Selection</h3>", unsafe_allow_html=True)
    possible_targets = [
        "Return on Ad Spend", "Return on Investment", "Profit Margin", 
        "Conversion Rate", "Click Through Rate", "Cost per Click", 
        "Cost per Acquisition", "Bounce Rate", "Average Order Value", 
        "Cost per Mille", "Customer Lifetime Value", "Budget Utilization", 
        "Churn Rate", "Retention Rate"
    ]
    
    selected_metrics = st.multiselect("Optimization Metrics", possible_targets, default=["Return on Ad Spend"])
    targets = {}
    if selected_metrics:
        for metric in selected_metrics:
            weight = st.slider(f"{metric}", 0.0, 1.0, 1.0, key=f"weight_{metric}")
            if weight > 0: targets[metric] = weight
    
    if not targets: targets = {"Return on Ad Spend": 1.0}
        
    st.markdown("---")
    max_iterations = st.slider("Max Iterations", 1, 5, 2)
    
    if st.button("🚀 Execute Optimization", use_container_width=True):
        if not st.session_state.campaigns:
            st.warning("Please add at least one campaign before optimizing.")
        else:
            with st.spinner("🤖 AI Agent performing deep analysis..."):
                progress_bar = st.progress(0)
                try:
                    campaign_inputs = [CampaignInput(**c) for c in st.session_state.campaigns]
                    progress_bar.progress(30)
                    state_input = {
                        "campaigns": campaign_inputs,
                        "iteration": 0,
                        "max_iterations": max_iterations,
                        "optimization_targets": targets
                    }
                    progress_bar.progress(50)
                    st.session_state.results = workflow.invoke(state_input)
                    save_state_to_disk(st.session_state.campaigns, st.session_state.results)
                    progress_bar.progress(100)
                    st.success("✅ Intelligence analysis complete!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Analysis Failed: {e}")

# Main Header
st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; font-size: 2.8rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;'>Marketing Intelligence Workspace</h1>
        <p style='margin:0.5rem 0 0 0; color: #475569; font-size: 1.2rem; font-weight: 500;'>Analyze, Optimize, and Visualize Global Marketing Strategy with AI</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 Campaign Orchestration", "📊 Performance Analytics", "📜 Strategic Blueprint"])

# Orchestration
with tab1:
    st.markdown("<h2 style='color: #1e293b; font-weight: 800;'>✨ Add New Campaign</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        with st.form("campaign_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                c_id = st.text_input("Campaign ID", placeholder="FB-SUM-XX")
                c_name = st.text_input("Campaign Name")
                channel = st.selectbox("Channel", ["Facebook", "Google Ads", "Instagram", "LinkedIn", "Twitter", "Email", "Other"])
            with col2:
                budget = st.number_input("Budget ($)", min_value=0.0)
                spend = st.number_input("Spend ($)", min_value=0.0)
                revenue = st.number_input("Revenue ($)", min_value=0.0)
            with col3:
                status = st.selectbox("Status", ["active", "paused", "completed", "archived"])
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")
            
            st.markdown("---")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1: impressions = st.number_input("Impressions", min_value=0, step=1000)
            with col_m2: clicks = st.number_input("Clicks", min_value=0, step=100)
            with col_m3: conversions = st.number_input("Conversions", min_value=0, step=10)
            with col_m4: bounces = st.number_input("Bounces", min_value=0, step=10)
            
            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
            with col_c1: leads = st.number_input("Leads", min_value=0, step=10)
            with col_c2: cust_start = st.number_input("Total Customers Start", min_value=0, step=10)
            with col_c3: cust_end = st.number_input("Total Customers End", min_value=0, step=10)
            with col_c4: 
                lost_cust = st.number_input("Lost Customers", min_value=0, step=1)
            
            avg_lifespan = st.number_input("Average Customer Lifespan (Months)", value=12.0, min_value=0.0, step=1.0)
                
            submitted = st.form_submit_button("➕ Add to Queue", use_container_width=True)
            if submitted:
                new_campaign = {
                    "campaign_id": c_id, "campaign_name": c_name, "channel": channel,
                    "budget": budget, "status": status, "impressions": impressions,
                    "clicks": clicks, "conversions": conversions, "spend": spend,
                    "revenue": revenue, "bounces": bounces, "total_customers_start": cust_start,
                    "total_customers_end": cust_end, "lost_customers": lost_cust, 
                    "avg_customer_lifespan": avg_lifespan,
                    "leads": leads, "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d")
                }
                try:
                    CampaignInput(**new_campaign)
                    st.session_state.campaigns.append(new_campaign)
                    save_state_to_disk(st.session_state.campaigns, st.session_state.results)
                    st.success(f"✅ Verified: {c_name} added to optimization queue.")
                except Exception as e:
                    st.error(f"❌ Validation Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.campaigns:
        st.markdown("<h3 style='color: #1e293b; font-weight: 700; margin-top: 2rem;'>📋 Current Campaign Queue</h3>", unsafe_allow_html=True)
        c_df = pd.DataFrame(st.session_state.campaigns)
        st.dataframe(c_df[["campaign_id", "campaign_name", "channel", "budget", "spend", "revenue"]], use_container_width=True)
        if st.button("🗑️ Clear Queue"):
            st.session_state.campaigns = []; st.session_state.results = None
            if os.path.exists(PERSISTENCE_FILE):
                try:
                    os.remove(PERSISTENCE_FILE)
                except Exception as e:
                    print(f"Error deleting file: {e}")
            st.rerun()

# Tab 2: Performance Analytics
with tab2:
    if st.session_state.results:
        results = st.session_state.results
        
        # Professional BI Header: Top-Level KPIs
        total_spend = sum(c.spend for c in results['campaigns'])
        total_revenue = sum(c.revenue for c in results['campaigns'])
        total_conversions = sum(c.conversions for c in results['campaigns'])
        total_impressions = sum(c.impressions for c in results['campaigns'])
        total_clicks = sum(c.clicks for c in results['campaigns'])
        avg_roas = total_revenue / total_spend if total_spend > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f'<div class="metric-card"><small>💰 Total Spend</small><h2>${total_spend:,.0f}</h2></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-card"><small>💵 Total Revenue</small><h2>${total_revenue:,.0f}</h2></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-card"><small>📈 Aggregate ROAS</small><h2>{avg_roas:.2f}x</h2></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="metric-card"><small>🎯 Conversions</small><h2>{total_conversions:,}</h2></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # PDF Export Button for Analytics
        analytics_pdf = generate_analytics_pdf(results)
        st.download_button(
            label="📄 Download Analytics Dashboard (PDF)",
            data=analytics_pdf,
            file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            key="download_analytics"
        )
        
        st.markdown("---")
        
        # Conversion Funnel
        st.markdown("<h3 style='color: #1e293b; font-weight: 700;'>🔄 Conversion Funnel Analysis</h3>", unsafe_allow_html=True)
        funnel_data = pd.DataFrame({
            'Stage': ['Impressions', 'Clicks', 'Conversions'],
            'Count': [total_impressions, total_clicks, total_conversions],
            'Percentage': [100, (total_clicks/total_impressions*100) if total_impressions > 0 else 0, 
                          (total_conversions/total_impressions*100) if total_impressions > 0 else 0]
        })
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_data['Stage'],
            x=funnel_data['Count'],
            textinfo="value+percent initial",
            marker=dict(color=['#667eea', '#764ba2', '#f093fb'])
        ))
        fig_funnel.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_funnel, use_container_width=True)
        
        st.markdown("---")
        
        # Efficiency & Reach
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("<h4 style='color: #1e293b; font-weight: 700;'>💎 Efficiency Frontier: Spend vs Revenue</h4>", unsafe_allow_html=True)
            chart_df = pd.DataFrame([{
                'Name': c.campaign_name, 'Spend': c.spend, 'Revenue': c.revenue, 'ROAS': c.revenue/c.spend if c.spend > 0 else 0
            } for c in results['campaigns']])
            fig_scatter = px.scatter(chart_df, x="Spend", y="Revenue", size="ROAS", color="ROAS",
                                    hover_name="Name", text="Name", template="plotly_white",
                                    color_continuous_scale='Purples')
            fig_scatter.update_traces(textposition='top center')
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col_right:
            st.markdown("<h4 style='color: #1e293b; font-weight: 700;'>🎨 Reach & Impact Bubble Chart</h4>", unsafe_allow_html=True)
            bubble_df = pd.DataFrame([{
                'Name': c.campaign_name, 'Impressions': c.impressions, 'Conversions': c.conversions, 'Spend': c.spend
            } for c in results['campaigns']])
            fig_bubble = px.scatter(bubble_df, x="Impressions", y="Conversions", size="Spend", color="Name",
                                   template="plotly_white")
            st.plotly_chart(fig_bubble, use_container_width=True)
            
        st.markdown("---")
        
        # Performance Quadrant
        st.markdown("<h3 style='color: #1e293b; font-weight: 700;'>📍 Performance Quadrant Matrix</h3>", unsafe_allow_html=True)
        quad_df = pd.DataFrame([{
            'Campaign': c.campaign_name,
            'ROAS': c.revenue/c.spend if c.spend > 0 else 0,
            'Spend': c.spend,
            'Channel': c.channel
        } for c in results['campaigns']])
        
        fig_quad = px.scatter(quad_df, x="Spend", y="ROAS", color="Channel", size="Spend",
                             hover_name="Campaign", template="plotly_white",
                             title="Campaign Performance Quadrant (High ROAS + High Spend = Stars)")
        
        # Add quadrant lines
        avg_spend = quad_df['Spend'].mean()
        avg_roas_quad = quad_df['ROAS'].mean()
        fig_quad.add_hline(y=avg_roas_quad, line_dash="dash", line_color="gray", annotation_text="Avg ROAS")
        fig_quad.add_vline(x=avg_spend, line_dash="dash", line_color="gray", annotation_text="Avg Spend")
        st.plotly_chart(fig_quad, use_container_width=True)
        
        st.markdown("---")
        
        # Comparison with Benchmarks
        st.markdown("<h3 style='color: #1e293b; font-weight: 700;'>📊 KPI Benchmark Variance Analysis</h3>", unsafe_allow_html=True)
        for i, insight in enumerate(results['insights']):
            with st.expander(f"🔍 Deep Dive: {insight['campaign_name']} ({insight['channel']})"):
                analysis_df = pd.DataFrame(insight['analysis']['kpi_analysis'])
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(name='Current', x=analysis_df['metric'], y=analysis_df['value'], 
                                         marker_color='#667eea'))
                fig_comp.add_trace(go.Bar(name='Benchmark', x=analysis_df['metric'], y=analysis_df['benchmark'], 
                                         marker_color='#cbd5e1'))
                fig_comp.update_layout(barmode='group', template="plotly_white", height=350)
                st.plotly_chart(fig_comp, use_container_width=True)
                st.table(analysis_df[['metric', 'value', 'benchmark', 'rating']])
                
        # Budget Shift Analysis
        st.markdown("<h3 style='color: #1e293b; font-weight: 700;'>💰 Budget Reallocation Strategy</h3>", unsafe_allow_html=True)
        recs = results.get('recommendations', [])
        if recs:
            rec_df = pd.DataFrame(recs)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name='Current', x=rec_df['campaign_name'], y=rec_df['current_budget'], 
                                    marker_color='#cbd5e1'))
            fig_bar.add_trace(go.Bar(name='AI Optimized', x=rec_df['campaign_name'], y=rec_df['recommended_budget'], 
                                    marker_color='#764ba2'))
            fig_bar.update_layout(barmode='group', template="plotly_white", 
                                 title="Current vs. AI Suggested Budgets", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("📥 Input campaign data and execute optimization to view comprehensive visual analytics.")

# Tab 3: Strategic Blueprint 
with tab3:
    if st.session_state.results:
        results = st.session_state.results
        
        col_rep, col_act = st.columns([2, 1])
        with col_rep:
            st.markdown("""
                <div style='background: rgba(255, 255, 255, 0.98); padding: 1.5rem; border-radius: 1rem; border: 2px solid #667eea; backdrop-filter: blur(10px); box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);'>
                    <h3 style='margin:0; color: #1e293b; font-weight: 800;'>📋 Executive Strategy Blueprint</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # PDF Export for Strategy
            strategy_txt = results.get('strategy_report', "")
            strategy_pdf = create_pdf_report("Marketing Strategic Blueprint", strategy_txt)
            st.download_button(
                label="📥 Export Report as PDF",
                data=strategy_pdf,
                file_name=f"strategy_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.write(strategy_txt)
            
        with col_act:
            st.markdown("<h4 style='color: #1e293b; font-weight: 700;'>⚠️ Risk Surveillance</h4>", unsafe_allow_html=True)
            for r in results.get('campaign_risks', []):
                risk_lvl = r['risk_level']
                s_color = "#ef4444" if risk_lvl=="High" else "#f59e0b" if risk_lvl=="Medium" else "#22c55e"
                st.markdown(f"""
                    <div style='border-left: 5px solid {s_color}; padding: 1rem; background: rgba(255,255,255,0.9); margin-bottom: 1rem; border-radius: 0.5rem;'>
                        <b style='color: #1e293b;'>{r['campaign_name']}</b><br>
                        <small style='color: {s_color}; font-weight: bold;'>Risk: {risk_lvl}</small><br>
                        <small>{', '.join(r['reasons'])}</small>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("<h4 style='color: #1e293b; font-weight: 700;'>🔄 Channel Optimization Pivots</h4>", unsafe_allow_html=True)
            for s in results.get('creative_channel_suggestions', []):
                with st.expander(f"💡 Pivot: {s['campaign_name']}"):
                    for sug in s['creative_channel_suggestions']:
                        st.markdown(f"• {sug}")
    else:
        st.info("🔮 Strategic insights will be generated upon execution.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.9rem;'>© 2026 AdAstra AI Intelligence | Precision Marketing Systems | Powered by LangGraph & Advanced AI</p>", unsafe_allow_html=True)
