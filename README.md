# AdAstra Intelligence: AI-Powered Marketing Optimization System 🎯

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Power-purple)](https://github.com/langchain-ai/langgraph)

**AdAstra Intelligence** is a state-of-the-art marketing analytics and optimization platform that leverages **LangGraph**, **Mathematical Optimization (CVXPY)**, and **Multi-Agent Orchestration** to transform raw campaign data into hyper-optimized marketing strategies.

Designed for high-performance marketing teams, AdAstra doesn't just show you what happened—it uses AI to tell you exactly how to reallocate your next $1,000,000 for maximum impact.

![Dashboard Preview](ai-marketing-campaign-optimization-logo.png)

---

## 🚀 Key Features

### 1. **Multi-Agent Optimization Workflow**
Built on a robust **LangGraph** architecture, the system orchestrates specialized AI agents:
- **Diagnostic Agent**: Compares campaign metrics against dynamically fetched industry benchmarks.
- **Risk Analyst Agent**: Heuristic and LLM-driven identification of high-risk/underperforming campaigns.
- **Sensitivity Agent**: Vectorized calculation of metric contributions to specify ROI drivers.
- **Creative Consultant**: Actionable suggestions for channel mix and asset improvements.

### 2. **Mathematical Budget Reallocation**
Unlike simple heuristics, AdAstra uses the **CVXPY** library with the **Clarabel solver** to perform constrained mathematical optimization. It reallocates your total budget across campaigns to maximize target KPIs while respecting real-world constraints (non-negativity, limited variance from historical spend).

### 3. **Breathtaking Analytics Dashboard**
A "premium" UI experience built with Streamlit and enhanced with custom CSS:
- **Glassmorphism UI**: Frosted glass cards, backdrop blurs, and animated gradients.
- **Interactive Visualizations**: Conversion Funnels, Performance Quadrants, Efficiency Frontiers, and Reach & Impact bubble charts.
- **Real-time Persistence**: Session state is automatically saved to disk, ensuring your analysis survives page reloads.

### 4. **Professional Executive Reporting**
Generate and download high-quality **PDF reports** powered by `ReportLab`. Reports include:
- **Global Performance Overview**: BI-grade KPI summaries.
- **Embedded Visualizations**: All dashboard charts are exported and embedded as high-resolution images.
- **Strategic Blueprints**: AI-generated executive summaries and tactical action plans.

---

## 🧠 The Optimization Workflow (LangGraph)

The system follows a sophisticated directed acyclic graph (DAG) with conditional looping:

1.  **Synchronization**: Normalizes user inputs and validates data schemas using Pydantic.
2.  **Benchmarking**: Dynamically generates channel-specific benchmarks using GPT-4o-mini.
3.  **Parallel Analysis (Map Stage)**: 
    *   Performance Diagnostics
    *   Risk Evaluation
    *   Metric Sensitivity Calculation
    *   Creative Asset Suggestions
4.  **Budget Optimization (Reduce Stage)**: Solves for the optimal budget distribution using mathematical constraints.
5.  **Strategy Synthesis**: Aggregates all agent outputs into a cohesive Markdown-based strategy report.
6.  **Conditional Refinement**: Evaluates results; if targets (e.g., ROAS >= 3.0) aren't met, it can loop back for further optimization.

![Workflow Architecture](marketing_campaign_optimization_workflow.png)

---

## 🛠️ Technical Stack

- **Frontend**: Streamlit (with advanced CSS/HTML injection)
- **Orchestration**: LangChain / LangGraph
- **Optimization**: CVXPY (Clarabel Solver), NumPy
- **Visuals**: Plotly Express/GraphObjects
- **Reporting**: ReportLab
- **Image Handling**: Kaleido / Cairo
- **Data Validation**: Pydantic v2
- **Persistence**: JSON-based local storage

---

## 📥 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/marketing-optimization-system.git
   cd marketing-optimization-system
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file or use Streamlit secrets:
   ```env
   OPENAI_API_KEY=your_key_here
   ```

---

## 🖥️ Usage Guide

### 1. **Campaign Input**
Navigate to the **Campaign Orchestration** tab. Enter your campaign metadata, financial performance (Spend/Revenue), and engagement metrics (Impressions/Conversions).

### 2. **AI Execution**
Use the sidebar to select your **Optimization Targets** (e.g., maximize ROAS, minimize CPA). Adjust metric weights to align with your business goals, then click **🚀 Execute Optimization**.

### 3. **Insight Analysis**
- View the **Performance Analytics** tab for multi-dimensional chart breakdowns.
- Review the **Strategic Blueprint** for the AI's tactical recommendations and risk surveillance.

### 4. **Reporting**
Download the **Analytics PDF** or the **Strategic Blueprint PDF** to share with stakeholders or prospective employers.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Sayam Kumar**  
*Data Scientist & AI Engineer*  
[LinkedIn](https://linkedin.com/in/sayamkumar) | [GitHub](https://github.com/sayamkumar)
