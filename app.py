import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(
    page_title="Matiks Gaming Analytics Dashboard",
    page_icon="🎮",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Matiks - Data Analyst Data - Sheet1.csv")
    
    # Convert date columns to datetime
    date_columns = ['Signup_Date', 'Last_Login']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%d-%b-%Y')
    
    return df

df = load_data()

# Title and description
st.title("🎮 Matiks Gaming Analytics Dashboard")
st.markdown("""
This dashboard provides insights into user behavior and revenue patterns for Matiks gaming platform.
Analyze daily active users, revenue trends, user segments, and identify opportunities for growth.
""")

# Sidebar for filtering
st.sidebar.header("Filters")

# Date range filter
min_date = df['Signup_Date'].min().date()
max_date = df['Last_Login'].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['Last_Login'].dt.date >= start_date) & 
                     (df['Last_Login'].dt.date <= end_date)]
else:
    filtered_df = df.copy()

# Game title filter
game_titles = ['All'] + sorted(df['Game_Title'].unique().tolist())
selected_game = st.sidebar.selectbox("Game Title", game_titles)

if selected_game != 'All':
    filtered_df = filtered_df[filtered_df['Game_Title'] == selected_game]

# Device type filter
device_types = ['All'] + sorted(df['Device_Type'].unique().tolist())
selected_device = st.sidebar.selectbox("Device Type", device_types)

if selected_device != 'All':
    filtered_df = filtered_df[filtered_df['Device_Type'] == selected_device]

# Main dashboard content
# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "👥 User Activity", 
    "💰 Revenue Analysis", 
    "🎯 User Segments", 
    "🔍 Insights"
])

# Tab 1: Overview
with tab1:
    st.header("Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_users = len(filtered_df)
        st.metric("Total Users", f"{total_users:,}")
    
    with col2:
        total_revenue = filtered_df['Total_Revenue_USD'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col3:
        avg_revenue_per_user = total_revenue / total_users if total_users > 0 else 0
        st.metric("Avg Revenue per User", f"${avg_revenue_per_user:.2f}")
    
    with col4:
        total_play_sessions = filtered_df['Total_Play_Sessions'].sum()
        st.metric("Total Play Sessions", f"{total_play_sessions:,}")
    
    # Overview charts
    st.subheader("User Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Device distribution
        device_counts = filtered_df['Device_Type'].value_counts().reset_index()
        device_counts.columns = ['Device Type', 'Count']
        
        fig = px.pie(
            device_counts, 
            values='Count', 
            names='Device Type',
            title='Users by Device Type',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Game distribution
        game_counts = filtered_df['Game_Title'].value_counts().reset_index()
        game_counts.columns = ['Game Title', 'Count']
        
        fig = px.pie(
            game_counts, 
            values='Count', 
            names='Game Title',
            title='Users by Game Title',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # User demographics
    st.subheader("User Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig = px.histogram(
            filtered_df,
            x='Age',
            nbins=20,
            title='Age Distribution',
            color_discrete_sequence=['#3366CC']
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_counts = filtered_df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        fig = px.bar(
            gender_counts,
            x='Gender',
            y='Count',
            title='Gender Distribution',
            color='Gender',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: User Activity
with tab2:
    st.header("User Activity Analysis")
    
    # Calculate DAU/WAU/MAU
    st.subheader("Daily, Weekly & Monthly Active Users")
    
    # Helper function to calculate active users
    def calculate_active_users(df, date_column, start_date, end_date, freq):
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        active_users = []
        
        if freq == 'D':  # Daily
            for date in date_range:
                count = len(df[df[date_column].dt.date == date.date()])
                active_users.append({'Date': date, 'Active Users': count})
        elif freq == 'W':  # Weekly
            for date in date_range:
                week_end = date
                week_start = week_end - timedelta(days=6)
                count = len(df[(df[date_column].dt.date >= week_start.date()) & 
                               (df[date_column].dt.date <= week_end.date())])
                active_users.append({'Date': date, 'Active Users': count})
        elif freq == 'M':  # Monthly
            for date in date_range:
                month_end = date
                month_start = date - pd.DateOffset(months=1) + pd.DateOffset(days=1)
                count = len(df[(df[date_column].dt.date >= month_start.date()) & 
                               (df[date_column].dt.date <= month_end.date())])
                active_users.append({'Date': date, 'Active Users': count})
        
        return pd.DataFrame(active_users)
    
    # Date range for active users calculation
    min_login = filtered_df['Last_Login'].min().date()
    max_login = filtered_df['Last_Login'].max().date()
    
    # Calculate DAU, WAU, MAU
    dau_df = calculate_active_users(filtered_df, 'Last_Login', min_login, max_login, 'D')
    wau_df = calculate_active_users(filtered_df, 'Last_Login', min_login, max_login, 'W')
    mau_df = calculate_active_users(filtered_df, 'Last_Login', min_login, max_login, 'M')
    
    # Create metrics for current DAU, WAU, MAU
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_dau = dau_df.iloc[-1]['Active Users'] if not dau_df.empty else 0
        st.metric("Current DAU", f"{current_dau:,}")
    
    with col2:
        current_wau = wau_df.iloc[-1]['Active Users'] if not wau_df.empty else 0
        st.metric("Current WAU", f"{current_wau:,}")
    
    with col3:
        current_mau = mau_df.iloc[-1]['Active Users'] if not mau_df.empty else 0
        st.metric("Current MAU", f"{current_mau:,}")
        
    # Stickiness ratio (DAU/MAU)
    if current_mau > 0:
        stickiness = current_dau / current_mau
        st.metric("Stickiness Ratio (DAU/MAU)", f"{stickiness:.2%}")
    
    # Plot DAU/WAU/MAU trends
    active_users_tab1, active_users_tab2 = st.tabs(["DAU/WAU/MAU Trends", "User Activity Patterns"])
    
    with active_users_tab1:
        # DAU chart
        st.subheader("Daily Active Users (DAU)")
        fig = px.line(
            dau_df, 
            x='Date', 
            y='Active Users',
            title='Daily Active Users Over Time',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # WAU chart
        st.subheader("Weekly Active Users (WAU)")
        fig = px.line(
            wau_df, 
            x='Date', 
            y='Active Users',
            title='Weekly Active Users Over Time',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # MAU chart
        st.subheader("Monthly Active Users (MAU)")
        fig = px.line(
            mau_df, 
            x='Date', 
            y='Active Users',
            title='Monthly Active Users Over Time',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with active_users_tab2:
        # User engagement metrics
        st.subheader("User Engagement Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            # Session frequency distribution
            fig = px.histogram(
                filtered_df,
                x='Total_Play_Sessions',
                nbins=20,
                title='Session Frequency Distribution',
                color_discrete_sequence=['#4CAF50']
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average session duration distribution
            fig = px.histogram(
                filtered_df,
                x='Avg_Session_Duration_Min',
                nbins=20,
                title='Average Session Duration (minutes)',
                color_discrete_sequence=['#FF9800']
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        # Total hours played distribution
        fig = px.histogram(
            filtered_df,
            x='Total_Hours_Played',
            nbins=30,
            title='Total Hours Played Distribution',
            color_discrete_sequence=['#2196F3']
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Early signs of churn - Time gaps analysis
        st.subheader("Potential Churn Indicators")
        
        # Calculate days since last login
        filtered_df['Days_Since_Last_Login'] = (pd.Timestamp.now() - filtered_df['Last_Login']).dt.days
        
        # Plot days since last login
        fig = px.histogram(
            filtered_df,
            x='Days_Since_Last_Login',
            nbins=20,
            title='Days Since Last Login',
            color_discrete_sequence=['#F44336']
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Identify users at risk of churning (not logged in for more than 30 days)
        churn_risk_df = filtered_df[filtered_df['Days_Since_Last_Login'] > 30].sort_values('Days_Since_Last_Login', ascending=False)
        
        if not churn_risk_df.empty:
            st.warning(f"⚠️ {len(churn_risk_df)} users haven't logged in for more than 30 days and may be at risk of churning.")
            
            # Show top 10 users at highest risk of churning
            st.dataframe(churn_risk_df[['Username', 'Days_Since_Last_Login', 'Total_Play_Sessions', 'Total_Revenue_USD', 'Subscription_Tier']].head(10))

# Tab 3: Revenue Analysis
with tab3:
    st.header("Revenue Analysis")
    
    # Key revenue metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_df['Total_Revenue_USD'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        avg_revenue_per_user = total_revenue / len(filtered_df) if len(filtered_df) > 0 else 0
        st.metric("Avg Revenue per User", f"${avg_revenue_per_user:.2f}")
    
    with col3:
        total_purchases = filtered_df['In_Game_Purchases_Count'].sum()
        st.metric("Total Purchases", f"{total_purchases:,}")
    
    with col4:
        avg_purchase_value = total_revenue / total_purchases if total_purchases > 0 else 0
        st.metric("Avg Purchase Value", f"${avg_purchase_value:.2f}")
    
    # Revenue trends over time
    st.subheader("Revenue Trends Over Time")
    
    # Group data by signup month for cohort analysis
    # Using 'M' despite the warning as 'ME' causes errors in this pandas version
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    filtered_df['Signup_Month'] = filtered_df['Signup_Date'].dt.to_period('M')
    monthly_revenue = filtered_df.groupby(filtered_df['Signup_Month'].dt.to_timestamp())['Total_Revenue_USD'].sum().reset_index()
    monthly_revenue.columns = ['Month', 'Revenue']
    
    # Plot monthly revenue trend
    fig = px.line(
        monthly_revenue,
        x='Month',
        y='Revenue',
        title='Monthly Revenue Trend',
        markers=True,
        labels={'Revenue': 'Revenue (USD)'},
        color_discrete_sequence=['#4CAF50']
    )
    fig.update_layout(xaxis_title='Month', yaxis_title='Revenue (USD)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Revenue breakdown by segments
    st.subheader("Revenue Breakdown by Segments")
    rev_tab1, rev_tab2, rev_tab3 = st.tabs(["By Device Type", "By Game Title", "By Subscription Tier"])
    
    with rev_tab1:
        # Revenue by device type
        device_revenue = filtered_df.groupby('Device_Type')['Total_Revenue_USD'].sum().reset_index()
        device_revenue.columns = ['Device Type', 'Revenue']
        
        fig = px.bar(
            device_revenue.sort_values('Revenue', ascending=False),
            x='Device Type',
            y='Revenue',
            title='Revenue by Device Type',
            color='Device Type',
            text_auto=True,
            labels={'Revenue': 'Revenue (USD)'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Average revenue per user by device type
        device_arpu = filtered_df.groupby('Device_Type').agg(
            Revenue=('Total_Revenue_USD', 'sum'),
            Users=('User_ID', 'count')
        ).reset_index()
        device_arpu['ARPU'] = device_arpu['Revenue'] / device_arpu['Users']
        
        fig = px.bar(
            device_arpu.sort_values('ARPU', ascending=False),
            x='Device_Type',
            y='ARPU',
            title='Average Revenue per User by Device Type',
            color='Device_Type',
            text_auto=True,
            labels={'ARPU': 'Avg Revenue per User (USD)'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with rev_tab2:
        # Revenue by game title
        game_revenue = filtered_df.groupby('Game_Title')['Total_Revenue_USD'].sum().reset_index()
        game_revenue.columns = ['Game Title', 'Revenue']
        
        fig = px.bar(
            game_revenue.sort_values('Revenue', ascending=False),
            x='Game Title',
            y='Revenue',
            title='Revenue by Game Title',
            color='Game Title',
            text_auto=True,
            labels={'Revenue': 'Revenue (USD)'},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Average revenue per user by game title
        game_arpu = filtered_df.groupby('Game_Title').agg(
            Revenue=('Total_Revenue_USD', 'sum'),
            Users=('User_ID', 'count')
        ).reset_index()
        game_arpu['ARPU'] = game_arpu['Revenue'] / game_arpu['Users']
        
        fig = px.bar(
            game_arpu.sort_values('ARPU', ascending=False),
            x='Game_Title',
            y='ARPU',
            title='Average Revenue per User by Game Title',
            color='Game_Title',
            text_auto=True,
            labels={'ARPU': 'Avg Revenue per User (USD)'},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with rev_tab3:
        # Revenue by subscription tier
        sub_revenue = filtered_df.groupby('Subscription_Tier')['Total_Revenue_USD'].sum().reset_index()
        sub_revenue.columns = ['Subscription Tier', 'Revenue']
        
        fig = px.bar(
            sub_revenue.sort_values('Revenue', ascending=False),
            x='Subscription Tier',
            y='Revenue',
            title='Revenue by Subscription Tier',
            color='Subscription Tier',
            text_auto=True,
            labels={'Revenue': 'Revenue (USD)'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # User count by subscription tier
        sub_users = filtered_df.groupby('Subscription_Tier')['User_ID'].count().reset_index()
        sub_users.columns = ['Subscription Tier', 'User Count']
        
        fig = px.bar(
            sub_users.sort_values('User Count', ascending=False),
            x='Subscription Tier',
            y='User Count',
            title='User Count by Subscription Tier',
            color='Subscription Tier',
            text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # High-value user analysis
    st.subheader("High-Value User Analysis")
    
    # Define high-value users (top 10% by revenue)
    revenue_threshold = filtered_df['Total_Revenue_USD'].quantile(0.9)
    high_value_users = filtered_df[filtered_df['Total_Revenue_USD'] >= revenue_threshold]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("High-Value Users", f"{len(high_value_users):,} ({len(high_value_users)/len(filtered_df):.1%} of total)")
    
    with col2:
        high_value_revenue = high_value_users['Total_Revenue_USD'].sum()
        high_value_revenue_pct = high_value_revenue / total_revenue if total_revenue > 0 else 0
        st.metric("Revenue from High-Value Users", f"${high_value_revenue:,.2f} ({high_value_revenue_pct:.1%} of total)")
    
    # High-value user characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        # Device type distribution for high-value users
        hv_device = high_value_users['Device_Type'].value_counts().reset_index()
        hv_device.columns = ['Device Type', 'Count']
        
        fig = px.pie(
            hv_device,
            values='Count',
            names='Device Type',
            title='High-Value Users by Device Type',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Game title distribution for high-value users
        hv_game = high_value_users['Game_Title'].value_counts().reset_index()
        hv_game.columns = ['Game Title', 'Count']
        
        fig = px.pie(
            hv_game,
            values='Count',
            names='Game Title',
            title='High-Value Users by Game Title',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Purchase frequency vs. revenue scatter plot
    # Make sure size values are positive
    size_values = filtered_df['Total_Hours_Played'].clip(lower=1)  # Ensure all values are at least 1
    
    fig = px.scatter(
        filtered_df,
        x='In_Game_Purchases_Count',
        y='Total_Revenue_USD',
        color='Device_Type',
        size=size_values,  # Use the clipped positive values
        hover_data=['Username', 'Game_Title', 'Subscription_Tier'],
        title='Purchase Frequency vs. Revenue',
        labels={
            'In_Game_Purchases_Count': 'Number of Purchases',
            'Total_Revenue_USD': 'Total Revenue (USD)',
            'Total_Hours_Played': 'Total Hours Played'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: User Segments
with tab4:
    st.header("User Segmentation Analysis")
    
    # User segmentation based on engagement and revenue
    st.subheader("User Segments by Engagement and Revenue")
    
    # Create a copy of the dataframe for segmentation
    segment_df = filtered_df.copy()
    
    # Normalize metrics for clustering
    from sklearn.preprocessing import StandardScaler
    
    # Select features for clustering
    features = ['Total_Play_Sessions', 'Avg_Session_Duration_Min', 'Total_Hours_Played', 'Total_Revenue_USD']
    scaler = StandardScaler()
    segment_df[features] = scaler.fit_transform(segment_df[features])
    
    # K-means clustering
    n_clusters = 4  # Number of clusters/segments
    
    # Add a slider to adjust number of clusters
    n_clusters = st.slider("Number of User Segments", min_value=2, max_value=6, value=4)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    filtered_df['Segment'] = kmeans.fit_predict(segment_df[features])
    filtered_df['Segment'] = 'Segment ' + (filtered_df['Segment'] + 1).astype(str)
    
    # Analyze segments
    segment_analysis = filtered_df.groupby('Segment').agg(
        Users=('User_ID', 'count'),
        Avg_Sessions=('Total_Play_Sessions', 'mean'),
        Avg_Duration=('Avg_Session_Duration_Min', 'mean'),
        Avg_Hours=('Total_Hours_Played', 'mean'),
        Avg_Revenue=('Total_Revenue_USD', 'mean'),
        Total_Revenue=('Total_Revenue_USD', 'sum')
    ).reset_index()
    
    # Format the metrics
    segment_analysis['Avg_Sessions'] = segment_analysis['Avg_Sessions'].round(1)
    segment_analysis['Avg_Duration'] = segment_analysis['Avg_Duration'].round(1)
    segment_analysis['Avg_Hours'] = segment_analysis['Avg_Hours'].round(1)
    segment_analysis['Avg_Revenue'] = segment_analysis['Avg_Revenue'].round(2)
    
    # Display segment analysis
    st.dataframe(segment_analysis, use_container_width=True)
    
    # Visualize segments
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot of segments by play time and revenue
        # Make sure size values are positive
        size_values = filtered_df['Total_Play_Sessions'].clip(lower=1)  # Ensure all values are at least 1
        
        fig = px.scatter(
            filtered_df,
            x='Total_Hours_Played',
            y='Total_Revenue_USD',
            color='Segment',
            size=size_values,  # Use the clipped positive values
            hover_data=['Username', 'Game_Title', 'Device_Type'],
            title='User Segments: Play Time vs. Revenue',
            labels={
                'Total_Hours_Played': 'Total Hours Played',
                'Total_Revenue_USD': 'Total Revenue (USD)',
                'Total_Play_Sessions': 'Number of Sessions'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Segment distribution
        segment_counts = filtered_df['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        fig = px.pie(
            segment_counts,
            values='Count',
            names='Segment',
            title='User Segment Distribution',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment characteristics
    st.subheader("Segment Characteristics")
    
    # Game preferences by segment
    game_segment = pd.crosstab(filtered_df['Segment'], filtered_df['Game_Title'])
    game_segment_pct = game_segment.div(game_segment.sum(axis=1), axis=0) * 100
    
    fig = px.imshow(
        game_segment_pct,
        labels=dict(x="Game Title", y="Segment", color="Percentage"),
        x=game_segment_pct.columns,
        y=game_segment_pct.index,
        color_continuous_scale='Viridis',
        title='Game Preferences by Segment (%)',
        text_auto=True
    )
    fig.update_traces(texttemplate='%{z:.1f}%', textfont={'size': 12})
    st.plotly_chart(fig, use_container_width=True)
    
    # Device preferences by segment
    device_segment = pd.crosstab(filtered_df['Segment'], filtered_df['Device_Type'])
    device_segment_pct = device_segment.div(device_segment.sum(axis=1), axis=0) * 100
    
    fig = px.imshow(
        device_segment_pct,
        labels=dict(x="Device Type", y="Segment", color="Percentage"),
        x=device_segment_pct.columns,
        y=device_segment_pct.index,
        color_continuous_scale='Viridis',
        title='Device Preferences by Segment (%)',
        text_auto=True
    )
    fig.update_traces(texttemplate='%{z:.1f}%', textfont={'size': 12})
    st.plotly_chart(fig, use_container_width=True)
    
    # Cohort Analysis
    st.subheader("Cohort Analysis")
    
    # Group users by signup month
    # Using 'M' despite the warning as 'ME' causes errors in this pandas version
    filtered_df['Signup_Month'] = filtered_df['Signup_Date'].dt.to_period('M')
    
    # Calculate cohort metrics
    cohort_data = filtered_df.groupby('Signup_Month').agg(
        Users=('User_ID', 'count'),
        Avg_Revenue=('Total_Revenue_USD', 'mean'),
        Total_Revenue=('Total_Revenue_USD', 'sum'),
        Avg_Sessions=('Total_Play_Sessions', 'mean'),
        Avg_Hours=('Total_Hours_Played', 'mean')
    ).reset_index()
    
    # Format the cohort data
    cohort_data['Signup_Month'] = cohort_data['Signup_Month'].dt.strftime('%Y-%m')
    cohort_data['Avg_Revenue'] = cohort_data['Avg_Revenue'].round(2)
    cohort_data['Avg_Sessions'] = cohort_data['Avg_Sessions'].round(1)
    cohort_data['Avg_Hours'] = cohort_data['Avg_Hours'].round(1)
    
    # Display cohort data
    st.dataframe(cohort_data, use_container_width=True)
    
    # Visualize cohort revenue trends
    fig = px.line(
        cohort_data,
        x='Signup_Month',
        y=['Avg_Revenue', 'Avg_Sessions', 'Avg_Hours'],
        title='Cohort Metrics Over Time',
        markers=True,
        labels={
            'value': 'Value',
            'Signup_Month': 'Signup Month',
            'variable': 'Metric'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: Insights
with tab5:
    st.header("Key Insights and Recommendations")
    
    # Summary of key findings
    st.subheader("Key Findings")
    
    # Calculate key metrics for insights
    total_users = len(filtered_df)
    total_revenue = filtered_df['Total_Revenue_USD'].sum()
    avg_revenue_per_user = total_revenue / total_users if total_users > 0 else 0
    
    # Top performing game
    game_revenue = filtered_df.groupby('Game_Title')['Total_Revenue_USD'].sum().reset_index()
    top_game = game_revenue.loc[game_revenue['Total_Revenue_USD'].idxmax()]['Game_Title']
    top_game_revenue = game_revenue['Total_Revenue_USD'].max()
    
    # Top performing device
    device_revenue = filtered_df.groupby('Device_Type')['Total_Revenue_USD'].sum().reset_index()
    top_device = device_revenue.loc[device_revenue['Total_Revenue_USD'].idxmax()]['Device_Type']
    top_device_revenue = device_revenue['Total_Revenue_USD'].max()
    
    # User retention metrics
    filtered_df['Days_Since_Last_Login'] = (pd.Timestamp.now() - filtered_df['Last_Login']).dt.days
    churn_risk_count = len(filtered_df[filtered_df['Days_Since_Last_Login'] > 30])
    churn_risk_pct = churn_risk_count / total_users if total_users > 0 else 0
    
    # High-value user metrics
    revenue_threshold = filtered_df['Total_Revenue_USD'].quantile(0.9)
    high_value_users = filtered_df[filtered_df['Total_Revenue_USD'] >= revenue_threshold]
    high_value_count = len(high_value_users)
    high_value_revenue = high_value_users['Total_Revenue_USD'].sum()
    high_value_revenue_pct = high_value_revenue / total_revenue if total_revenue > 0 else 0
    
    # Display insights
    st.info(f"📊 **User Base**: The platform has {total_users:,} users generating ${total_revenue:,.2f} in total revenue (${avg_revenue_per_user:.2f} per user).")
    
    st.info(f"🎮 **Top Game**: '{top_game}' is the highest-revenue generating game with ${top_game_revenue:,.2f} in total revenue.")
    
    st.info(f"📱 **Top Device**: '{top_device}' users generate the most revenue with ${top_device_revenue:,.2f} in total.")
    
    st.info(f"💰 **Revenue Distribution**: {high_value_count} high-value users ({high_value_count/total_users:.1%} of user base) generate ${high_value_revenue:,.2f} ({high_value_revenue_pct:.1%} of total revenue).")
    
    st.warning(f"⚠️ **Churn Risk**: {churn_risk_count} users ({churn_risk_pct:.1%} of user base) haven't logged in for over 30 days and are at risk of churning.")
    
    # Recommendations
    st.subheader("Recommendations")
    
    st.success("""
    ### 1. Retention Strategy
    - **Implement re-engagement campaigns** targeting users who haven't logged in for 15+ days
    - **Create special events or bonuses** for returning users to reduce churn rate
    - **Develop a loyalty program** that rewards consistent play patterns
    """)
    
    st.success("""
    ### 2. Revenue Optimization
    - **Focus on converting Free tier users** to paid subscription tiers with targeted promotions
    - **Optimize in-game purchase offerings** based on popular items among high-value users
    - **Create special bundles** for the most popular game titles to increase monetization
    """)
    
    st.success("""
    ### 3. Platform Development
    - **Prioritize development for top-performing devices** to maximize ROI
    - **Enhance features for the highest-revenue game titles** to further boost engagement
    - **Improve onboarding experience** to increase early user retention and conversion
    """)
    
    st.success("""
    ### 4. User Segmentation Strategy
    - **Create personalized experiences** for different user segments based on their behavior patterns
    - **Develop targeted marketing campaigns** for each segment with relevant offers
    - **Focus resources on high-potential segments** that show growth in both engagement and spending
    """)
    
    # Next steps
    st.subheader("Next Steps for Analysis")
    
    st.info("""
    1. **Conduct A/B testing** on different monetization strategies for key user segments
    2. **Implement more detailed cohort tracking** to better understand user lifecycle
    3. **Analyze user journey and conversion funnels** to identify optimization opportunities
    4. **Collect additional data points** on user satisfaction and feature preferences
    5. **Set up real-time monitoring** of key metrics to enable faster response to trends
    """)
    
    # Methodology note
    st.caption("""
    **Methodology Note**: This analysis is based on the provided dataset and focuses on user behavior, revenue patterns, and segmentation.
    The dashboard uses various analytical techniques including cohort analysis, clustering for segmentation, and trend analysis to derive insights.
    Recommendations are data-driven and aim to improve user retention, engagement, and revenue generation.
    """)

