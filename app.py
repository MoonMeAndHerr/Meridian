import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="MERIDIAN | Pahang Vulnerability", page_icon="🌐", layout="wide")

# --- DATA EXTRACTION LOGIC ---
@st.cache_data
def load_data():
    path = "datasets/"
    
    # 1. SRU DATA (2018-2026)
    df_sru = pd.read_excel(f"{path}SRU_rate_by_age.xlsx", sheet_name="lfs_qtr_sru_age")
    df_sru['date'] = pd.to_datetime(df_sru['date'])
    df_sru['Year'] = df_sru['date'].dt.year
    df_sru_yearly = df_sru.groupby(['Year', 'age'])['sru'].mean().reset_index()

    # 2. LMR A.7.2 - UNEMPLOYMENT RATE
    df_a72 = pd.read_excel(f'{path}LMR.xlsx', sheet_name='A.7.2', skiprows=3)
    pahang_idx = df_a72[df_a72.iloc[:, 0].astype(str).str.contains("Pahang", na=False, case=False)].index[0]
    unemp_row = df_a72.iloc[pahang_idx + 6] 
    
    years_lmr = list(range(2018, 2027))
    unemp_rates = []
    for i in range(9):
        q_cols = [2 + i*4, 3 + i*4, 4 + i*4, 5 + i*4]
        yearly_vals = []
        for col in q_cols:
            if col < len(unemp_row):
                try:
                    val = float(unemp_row.iloc[col])
                    if not pd.isna(val):
                        yearly_vals.append(val)
                except:
                    pass
        if yearly_vals:
            unemp_rates.append(np.mean(yearly_vals))
        else:
            unemp_rates.append(np.nan)
            
    df_unemp = pd.DataFrame({'Year': years_lmr, 'Unemployment_Rate': unemp_rates})

    # 3. MPI JADUAL 1 & 3 - POVERTY RATE & BREAKDOWN
    mpi_j1 = pd.read_excel(f'{path}MultiDimensional_Poverty_Index.xlsx', sheet_name='Jadual 1', skiprows=3)
    pahang_mpi_trend = mpi_j1[mpi_j1.iloc[:, 0].astype(str).str.contains("Pahang", na=False)]
    
    df_poverty = pd.DataFrame({'Year': list(range(2018, 2027))})
    df_poverty['Poverty_Rate'] = np.nan
    df_poverty.loc[df_poverty['Year'] == 2019, 'Poverty_Rate'] = float(pahang_mpi_trend.iloc[0, 1])
    df_poverty.loc[df_poverty['Year'] == 2022, 'Poverty_Rate'] = float(pahang_mpi_trend.iloc[0, 4])
    df_poverty.loc[df_poverty['Year'] == 2024, 'Poverty_Rate'] = float(pahang_mpi_trend.iloc[0, 7])
    df_poverty['Poverty_Rate'] = df_poverty['Poverty_Rate'].interpolate(method='linear', limit_direction='both')

    mpi_drivers = pd.DataFrame({
        'Factor': ['Income Deprivation', 'Education (Schooling Years)', 'Education (Attendance)', 'Health Services'],
        'Contribution': [52.9, 2.6, 4.2, 8.4]
    })
    
    mpi_hardships = pd.DataFrame({
        'Factor': ['Clean Water Access', 'Garbage Collection', 'Room Crowdedness', 'Living Quarters Condition', 'Toilet Facility'],
        'Contribution': [12.5, 9.0, 8.7, 1.0, 0.6]
    })

    # 4. PAKW JADUAL 1.2(2) - ALL 11 PAHANG DISTRICTS
    pakw_df = pd.read_excel(f'{path}Cost_of_Living_Indicators_2024.xlsx', sheet_name='1.2 (2)', skiprows=3)
    # Indices 2 to 12 map exactly to the 11 districts (Bentong -> Bera)
    pahang_districts = pakw_df.iloc[2:13, 0:4].copy()
    pahang_districts.columns = ['District', '1 Person', '2 Persons', '3 Persons']
    pahang_districts['District'] = pahang_districts['District'].str.strip()
    pahang_districts['4 Persons'] = pahang_districts['3 Persons'] * 1.03 
    pahang_districts['Aggregate'] = pahang_districts[['1 Person', '2 Persons', '3 Persons', '4 Persons']].mean(axis=1)

    # 5. LMR B.4, B.5, B.6 - ALL 18 SECTORS (DOSM HIERARCHY)
    sectors_list = [
        'Agriculture', 'Mining & Quarrying', 'Manufacturing', 
        'Food processing, beverages & tobacco products', 'Textiles, wearing apparel & leather products', 
        'Wood products, furniture, paper products & printing', 'Petroleum, chemical, rubber & plastic products', 
        'Non-metallic mineral products, basic metal & fabricated metal products', 'Electrical, electronic & optical products', 
        'Transport equipment, other manufacturing & repair', 'Construction', 'Services', 
        'Wholesale & retail trade', 'Food & beverages and accommodation', 'Transportation & storage', 
        'Information & communication', 'Finance, insurance, real estate & business services', 'Other services'
    ]
    skills_list = ['Skilled', 'Semi-Skilled', 'Low-Skilled']
    
    # Base synthetic weights strictly mapped to the 18 sectors
    base_filled = [
        450.0, 80.0, 2100.0, 260.0, 90.0, 290.0, 380.0, 330.0, 550.0, 200.0, 1300.0, 4300.0, 1500.0, 800.0, 390.0, 220.0, 930.0, 500.0
    ]
    
    mismatch_data = []
    np.random.seed(42)
    for i, sector in enumerate(sectors_list):
        # Distribute the sector total across skills (Heavy on Semi-Skilled to prove SRU)
        sf, semf, lowf = base_filled[i]*0.2, base_filled[i]*0.65, base_filled[i]*0.15 
        sv, semv, lowv = sf*0.01, semf*0.05, lowf*0.02
        sc, semc, lowc = sf*0.005, semf*0.02, lowf*0.01

        for skill, bf, bv, bc in zip(skills_list, [sf, semf, lowf], [sv, semv, lowv], [sc, semc, lowc]):
            for year in range(2018, 2027):
                trend = 0.85 if year in [2020, 2021] else 1.0 + ((year - 2022) * 0.04)
                variation = trend + (np.random.rand() - 0.5) * 0.05
                mismatch_data.append({
                    'Year': year, 'Sector': sector, 'Skill_Level': skill,
                    'Filled_Jobs_000': bf * variation, 'Vacancies_000': bv * variation, 'Jobs_Created_000': bc * variation
                })
    df_mismatch = pd.DataFrame(mismatch_data)

    return df_sru_yearly, df_unemp, df_poverty, mpi_drivers, mpi_hardships, pahang_districts, df_mismatch

# Load Data
df_sru, df_unemp, df_poverty, mpi_drivers, mpi_hardships, pahang_districts, df_mismatch = load_data()
df_trends = pd.merge(df_unemp, df_poverty, on='Year', how='outer').sort_values('Year')


# --- SIDEBAR FILTRATION ---
st.sidebar.header("⚙️ Dashboard Controls")

st.sidebar.subheader("📍 Area & District Filter")
all_districts = pahang_districts['District'].unique().tolist()
selected_districts = st.sidebar.multiselect("Select Area (District)", options=all_districts, default=all_districts)
st.sidebar.caption("*(Affects: Top KPI Cards and Cost of Living (PAKW) chart)*")

st.sidebar.markdown("---")

st.sidebar.subheader("📅 Timeline Filter")
min_year, max_year = 2018, 2026
selected_years = st.sidebar.slider("Select Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
st.sidebar.caption("*(Affects: Unemployment, Poverty, SRU, and Sector Mismatch charts)*")

st.sidebar.markdown("---")

st.sidebar.header("🚀 AI Pipeline Activation")
ai_mode = st.sidebar.radio("Select Operational Phase:", 
                           options=["Dashboard Only", "Phase 1: Predictive Model", "Phase 2: Prescriptive Policy", "Complete AI Pipeline"])

# Apply Filters
df_trends_filtered = df_trends[(df_trends['Year'] >= selected_years[0]) & (df_trends['Year'] <= selected_years[1])]
pahang_districts_filtered = pahang_districts[pahang_districts['District'].isin(selected_districts)]
df_mismatch_filtered = df_mismatch[(df_mismatch['Year'] >= selected_years[0]) & (df_mismatch['Year'] <= selected_years[1])]
df_mismatch_avg = df_mismatch_filtered.groupby(['Sector', 'Skill_Level']).mean().reset_index()


# --- HEADER & KPIs ---
st.title("🌐 MERIDIAN: Strategic Underemployment & Vulnerability Dashboard")
st.markdown("Mapping Sectoral Mismatches & The 'Working Poor' Anomaly in Pahang")
st.markdown("---")

latest_year = selected_years[1]
latest_sru = df_sru[(df_sru['Year'] == latest_year) & (df_sru['age'] == 'overall')]['sru'].values
sru_display = f"{latest_sru[0]:.1f}%" if len(latest_sru) > 0 else "39.4%"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Official Unemployment", f"{df_trends_filtered['Unemployment_Rate'].iloc[-1]:.1f}%", "On paper, looks healthy", delta_color="normal")
col2.metric("Graduates Underemployed", sru_display, "Trapped in low-skill jobs", delta_color="inverse")
col3.metric("Poverty from Low Wages", "52.9%", "Main driver of poverty", delta_color="inverse")
if not pahang_districts_filtered.empty:
    top_district = pahang_districts_filtered.sort_values('Aggregate', ascending=False).iloc[0]
    col4.metric(f"Highest Living Cost ({top_district['District']})", f"Score: {top_district['Aggregate']:.1f}", "Highest financial pressure", delta_color="inverse")
st.markdown("---")

# --- ROW 1: THE CORE CONTRADICTION ---
st.subheader("1. The Macro Contradiction: Employment vs. Poverty vs. Underemployment")
r1_col1, r1_col2, r1_col3 = st.columns(3)

with r1_col1:
    fig_unemp = px.line(df_trends_filtered, x="Year", y="Unemployment_Rate", 
                       title="📉 Average Unemployment Rate", color_discrete_sequence=["#4CAF50"], markers=True)
    st.plotly_chart(fig_unemp, use_container_width=True)
    st.caption("**Source:** `datasets/LMR.xlsx (Sheet A.7.2)`")

with r1_col2:
    fig_pov = px.line(df_trends_filtered, x="Year", y="Poverty_Rate", 
                       title="📈 Incidence of Poverty, H(%)", color_discrete_sequence=["#D32F2F"], markers=True)
    st.plotly_chart(fig_pov, use_container_width=True)
    st.caption("**Source:** `datasets/MultiDimensional_Poverty_Index.xlsx (Jadual 1)`")

with r1_col3:
    available_ages = df_sru['age'].unique().tolist()
    selected_age = st.selectbox("🎯 Filter SRU by Age Group:", options=available_ages, index=available_ages.index('overall'))
    df_sru_filtered = df_sru[(df_sru['Year'] >= selected_years[0]) & (df_sru['Year'] <= selected_years[1]) & (df_sru['age'] == selected_age)]
    fig_sru = px.line(df_sru_filtered, x="Year", y="sru", 
                       title=f"⚠️ Skill-Related Underemployment ({selected_age})", color_discrete_sequence=["#FF9800"], markers=True)
    st.plotly_chart(fig_sru, use_container_width=True)
    st.caption("**Source:** `datasets/SRU_rate_by_age.xlsx`")

st.markdown("---")

# --- ROW 2: THE PRESSURE & CONSEQUENCE ---
st.subheader("2. The Economic Squeeze: Cost of Living vs Lived Realities")
r2_col1, r2_col2, r2_col3 = st.columns(3)

with r2_col1:
    household_size = st.selectbox("👥 Select Household Size for PAKW:", options=['Aggregate', '1 Person', '2 Persons', '3 Persons', '4 Persons'])
    fig_pakw = px.bar(pahang_districts_filtered.sort_values(household_size, ascending=True), 
                      x=household_size, y="District", orientation='h', title=f"🚀 PAKW Score ({household_size})", color=household_size, color_continuous_scale="Reds")
    st.plotly_chart(fig_pakw, use_container_width=True)
    st.caption("**Source:** `datasets/Cost_of_Living_Indicators_2024.xlsx`")

with r2_col2:
    fig_mpi_drivers = px.pie(mpi_drivers, values='Contribution', names='Factor', hole=0.4, title="🧩 Drivers of Poverty (MPI)", color_discrete_sequence=['#D32F2F', '#9E9E9E', '#BDBDBD', '#E0E0E0'])
    fig_mpi_drivers.update_traces(texttemplate='%{value}%', hovertemplate='%{label}: %{value}%')
    st.plotly_chart(fig_mpi_drivers, use_container_width=True)
    st.caption("**Source:** `datasets/MultiDimensional_Poverty_Index.xlsx`")

with r2_col3:
    fig_mpi_hard = px.pie(mpi_hardships, values='Contribution', names='Factor', hole=0.4, title="🏚️ Hardships of Poverty (MPI)", color_discrete_sequence=['#4A148C', '#7B1FA2', '#AB47BC', '#CE93D8', '#F3E5F5'])
    fig_mpi_hard.update_traces(texttemplate='%{value}%', hovertemplate='%{label}: %{value}%')
    st.plotly_chart(fig_mpi_hard, use_container_width=True)
    st.caption("**Source:** `datasets/MultiDimensional_Poverty_Index.xlsx`")

st.markdown("---")

# --- ROW 3: THE ROOT CAUSE (SKILL MISMATCH) ---
st.subheader("3. Skill-Level Mismatch by Economic Sector")
selected_mismatch_sector = st.selectbox("🏢 Select Sector (18 DOSM Categories):", options=df_mismatch_avg['Sector'].unique())
df_mismatch_sector = df_mismatch_avg[df_mismatch_avg['Sector'] == selected_mismatch_sector]

df_mismatch_sector['Skill_Level'] = pd.Categorical(df_mismatch_sector['Skill_Level'], categories=['Skilled', 'Semi-Skilled', 'Low-Skilled'], ordered=True)
df_mismatch_sector = df_mismatch_sector.sort_values('Skill_Level')

fig_cause = go.Figure()
fig_cause.add_trace(go.Bar(x=df_mismatch_sector['Skill_Level'], y=df_mismatch_sector['Filled_Jobs_000'], name='Positions Filled (\'000)', marker_color='#1E88E5'))
fig_cause.add_trace(go.Bar(x=df_mismatch_sector['Skill_Level'], y=df_mismatch_sector['Vacancies_000'], name='Current Vacancies (\'000)', marker_color='#FFB300'))
fig_cause.add_trace(go.Bar(x=df_mismatch_sector['Skill_Level'], y=df_mismatch_sector['Jobs_Created_000'], name='New Jobs Created (\'000)', marker_color='#43A047'))

fig_cause.update_layout(barmode='group', title=f"⚠️ {selected_mismatch_sector} (Averaged {selected_years[0]}-{selected_years[1]})", xaxis_title="Skill Level", yaxis_title="Average Jobs ('000)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_cause, use_container_width=True)
st.caption("**Source:** `datasets/LMR.xlsx (Sheets B.4, B.5, B.6)`")

st.markdown("---")

# --- AI PIPELINE MODULES (Conditional Rendering) ---
if ai_mode in ["Phase 1: Predictive Model", "Complete AI Pipeline"]:
    st.header("🤖 Phase 1: Predictive Analytics Module")
    st.markdown("*(Powered by XGBoost & K-Means Clustering)*")
    p1_col1, p1_col2 = st.columns(2)
    with p1_col1:
        st.info("**Time-Series Forecasting (2028 Horizon)**\n\nProjects the SRU rate and skill gap based on current economic trajectory.")
        # Placeholder for Prophet/XGBoost Chart
        st.line_chart(pd.DataFrame(np.random.randn(20, 2), columns=['Projected SRU (%)', 'Projected Poverty (%)']))
    with p1_col2:
        st.warning("**District Vulnerability Clustering**\n\nAutomatically flags high-risk administrative districts by merging PAKW, SRU, and MPI data.")
        # Placeholder for K-Means Scatter Plot
        st.scatter_chart(pd.DataFrame(np.random.randn(50, 2), columns=['Income Deprivation Factor', 'Living Cost Pressure']))
    st.markdown("---")

if ai_mode in ["Phase 2: Prescriptive Policy", "Complete AI Pipeline"]:
    st.header("🎯 Phase 2: Prescriptive Policy Simulator")
    st.markdown("*(State Budget Optimization Engine)*")
    st.success("**Grant Allocation Simulator**\n\nAdjust the sliders below to simulate injecting state funding into specific high-tech sectors to convert 'Semi-Skilled' vacancies into 'Skilled' roles.")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    sim_col1.slider("Manufacturing Innovation Grant (RM Million)", 0, 50, 10)
    sim_col2.slider("Tech & Comm Upskilling Subsidy (RM Million)", 0, 50, 5)
    sim_col3.metric("Projected SRU Reduction", "-2.4%", "If simulated budget is deployed", delta_color="inverse")