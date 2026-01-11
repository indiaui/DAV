import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Data Visualization Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Section headers */
    .section-header {
        color: #1a1a2e;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Plot container */
    .plot-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        background-color: #f0f2f6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #667eea;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Data Visualization Studio</h1>
    <p>Create stunning visualizations from your data with ease</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'plot_generated' not in st.session_state:
    st.session_state.plot_generated = False

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Upload Your Data")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset to start visualizing"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.success(f"✅ Loaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        
        st.metric("Missing Values", df.isnull().sum().sum())
        
        # Column types
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        cat_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        
        st.markdown(f"**Numeric Columns:** {num_cols}")
        st.markdown(f"**Categorical Columns:** {cat_cols}")

# Main content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Data Preview Section
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("**📊 Statistical Summary**")
            st.dataframe(df.describe(), use_container_width=True)
        with col_info2:
            st.markdown("**📝 Column Information**")
            col_info = pd.DataFrame({
                'Type': df.dtypes,
                'Non-Null': df.count(),
                'Null': df.isnull().sum(),
                'Unique': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
    
    # Plot Configuration
    st.markdown('<p class="section-header">🎨 Plot Configuration</p>', unsafe_allow_html=True)
    
    # Plot categories
    plot_categories = {
        "📊 Basic Plots": {
            "Line Plot": "line",
            "Bar Chart": "bar",
            "Scatter Plot": "scatter",
            "Area Chart": "area",
            "Pie Chart": "pie"
        },
        "📈 Statistical Plots": {
            "Histogram": "histogram",
            "Box Plot": "box",
            "Violin Plot": "violin",
            "KDE Plot": "kde",
            "ECDF Plot": "ecdf"
        },
        "🔗 Relationship Plots": {
            "Correlation Heatmap": "heatmap",
            "Pair Plot": "pairplot",
            "Joint Plot": "jointplot",
            "Regression Plot": "regplot",
            "Scatter Matrix": "scatter_matrix"
        },
        "📉 Distribution Plots": {
            "Dist Plot": "distplot",
            "Rug Plot": "rugplot",
            "Strip Plot": "stripplot",
            "Swarm Plot": "swarmplot"
        },
        "🎯 Categorical Plots": {
            "Count Plot": "countplot",
            "Cat Plot": "catplot",
            "Point Plot": "pointplot",
            "Bar Plot (Seaborn)": "barplot_sns"
        }
    }
    
    # Create columns for plot selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        category = st.selectbox(
            "📂 Select Plot Category",
            options=list(plot_categories.keys()),
            help="Choose a category of plots"
        )
    
    with col2:
        plot_type = st.selectbox(
            "📊 Select Plot Type",
            options=list(plot_categories[category].keys()),
            help="Choose the specific plot you want to create"
        )
    
    plot_key = plot_categories[category][plot_type]
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df.columns.tolist()
    
    st.markdown("---")
    st.markdown('<p class="section-header">⚙️ Feature Selection</p>', unsafe_allow_html=True)
    
    # Dynamic feature selection based on plot type
    x_col, y_col, hue_col = None, None, None
    
    # Feature selection columns
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    # Plots that need X and Y
    xy_plots = ['line', 'scatter', 'bar', 'area', 'regplot', 'jointplot']
    # Plots that need only X
    x_only_plots = ['histogram', 'kde', 'distplot', 'rugplot', 'ecdf', 'pie', 'countplot']
    # Plots that can use hue
    hue_plots = ['scatter', 'line', 'histogram', 'kde', 'box', 'violin', 'stripplot', 'swarmplot', 'barplot_sns', 'pointplot']
    
    with feat_col1:
        if plot_key in xy_plots + x_only_plots + ['box', 'violin', 'stripplot', 'swarmplot', 'barplot_sns', 'pointplot']:
            if plot_key in ['countplot'] + ['box', 'violin', 'stripplot', 'swarmplot']:
                x_col = st.selectbox("🎯 X-Axis (Feature)", options=all_cols, key="x_col")
            else:
                x_col = st.selectbox("🎯 X-Axis (Feature)", options=numeric_cols if numeric_cols else all_cols, key="x_col")
    
    with feat_col2:
        if plot_key in xy_plots + ['box', 'violin', 'stripplot', 'swarmplot', 'barplot_sns', 'pointplot']:
            y_col = st.selectbox("📊 Y-Axis (Feature)", options=numeric_cols if numeric_cols else all_cols, key="y_col")
    
    with feat_col3:
        if plot_key in hue_plots:
            hue_options = ["None"] + categorical_cols
            hue_selection = st.selectbox("🎨 Hue (Optional)", options=hue_options, key="hue_col")
            hue_col = None if hue_selection == "None" else hue_selection
    
    # Additional plot settings
    st.markdown("---")
    st.markdown('<p class="section-header">🎛️ Plot Customization</p>', unsafe_allow_html=True)
    
    setting_col1, setting_col2, setting_col3, setting_col4 = st.columns(4)
    
    with setting_col1:
        # Color palette selection
        palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind', 
                   'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                   'coolwarm', 'RdYlBu', 'RdYlGn', 'Spectral']
        palette = st.selectbox("🎨 Color Palette", options=palettes)
    
    with setting_col2:
        # Figure size
        fig_width = st.slider("📏 Width", min_value=6, max_value=20, value=10)
    
    with setting_col3:
        fig_height = st.slider("📐 Height", min_value=4, max_value=15, value=6)
    
    with setting_col4:
        # Style
        styles = ['whitegrid', 'darkgrid', 'white', 'dark', 'ticks']
        style = st.selectbox("🖼️ Style", options=styles)
    
    # Additional options row
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    
    with opt_col1:
        plot_title = st.text_input("📝 Plot Title", value=f"{plot_type}")
    
    with opt_col2:
        x_label = st.text_input("🏷️ X-Axis Label", value=x_col if x_col else "")
    
    with opt_col3:
        y_label = st.text_input("🏷️ Y-Axis Label", value=y_col if y_col else "")
    
    # Special options for specific plots
    extra_options = {}
    
    if plot_key == 'histogram':
        hist_col1, hist_col2 = st.columns(2)
        with hist_col1:
            extra_options['bins'] = st.slider("📊 Number of Bins", 5, 100, 30)
        with hist_col2:
            extra_options['kde_overlay'] = st.checkbox("Show KDE Overlay", value=True)
    
    if plot_key == 'scatter':
        scatter_col1, scatter_col2 = st.columns(2)
        with scatter_col1:
            extra_options['size'] = st.slider("⚫ Point Size", 10, 200, 50)
        with scatter_col2:
            extra_options['alpha'] = st.slider("🔍 Transparency", 0.1, 1.0, 0.7)
    
    if plot_key in ['pairplot', 'scatter_matrix']:
        cols_for_pair = st.multiselect(
            "Select Columns for Pair Plot",
            options=numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))]
        )
        extra_options['pair_cols'] = cols_for_pair
    
    st.markdown("---")
    
    # Plot button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        generate_plot = st.button("🚀 Generate Plot", use_container_width=True)
    
    # Generate and display plot
    if generate_plot:
        st.markdown("---")
        st.markdown('<p class="section-header">📊 Your Visualization</p>', unsafe_allow_html=True)
        
        try:
            # Set style
            sns.set_style(style)
            sns.set_palette(palette)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Generate plot based on type
            if plot_key == 'line':
                if hue_col:
                    sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
                else:
                    sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
            
            elif plot_key == 'bar':
                if len(df[x_col].unique()) > 20:
                    top_categories = df[x_col].value_counts().head(20).index
                    plot_df = df[df[x_col].isin(top_categories)]
                else:
                    plot_df = df
                plot_df.groupby(x_col)[y_col].mean().plot(kind='bar', ax=ax, color=sns.color_palette(palette)[0])
                plt.xticks(rotation=45, ha='right')
            
            elif plot_key == 'scatter':
                scatter_params = {'data': df, 'x': x_col, 'y': y_col, 'ax': ax}
                if hue_col:
                    scatter_params['hue'] = hue_col
                scatter_params['s'] = extra_options.get('size', 50)
                scatter_params['alpha'] = extra_options.get('alpha', 0.7)
                sns.scatterplot(**scatter_params)
            
            elif plot_key == 'area':
                df_sorted = df.sort_values(x_col)
                ax.fill_between(df_sorted[x_col], df_sorted[y_col], alpha=0.7)
                ax.plot(df_sorted[x_col], df_sorted[y_col])
            
            elif plot_key == 'pie':
                plt.close()
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                value_counts = df[x_col].value_counts()
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                value_counts.plot.pie(ax=ax, autopct='%1.1f%%', colors=sns.color_palette(palette), startangle=90)
                ax.set_ylabel('')
            
            elif plot_key == 'histogram':
                hist_params = {'data': df, 'x': x_col, 'ax': ax, 'bins': extra_options.get('bins', 30)}
                if extra_options.get('kde_overlay', True):
                    hist_params['kde'] = True
                if hue_col:
                    hist_params['hue'] = hue_col
                sns.histplot(**hist_params)
            
            elif plot_key == 'box':
                box_params = {'data': df, 'x': x_col, 'ax': ax}
                if y_col:
                    box_params['y'] = y_col
                if hue_col:
                    box_params['hue'] = hue_col
                sns.boxplot(**box_params)
                plt.xticks(rotation=45, ha='right')
            
            elif plot_key == 'violin':
                violin_params = {'data': df, 'x': x_col, 'ax': ax}
                if y_col:
                    violin_params['y'] = y_col
                if hue_col:
                    violin_params['hue'] = hue_col
                sns.violinplot(**violin_params)
                plt.xticks(rotation=45, ha='right')
            
            elif plot_key == 'kde':
                kde_params = {'data': df, 'x': x_col, 'ax': ax, 'fill': True}
                if hue_col:
                    kde_params['hue'] = hue_col
                sns.kdeplot(**kde_params)
            
            elif plot_key == 'ecdf':
                ecdf_params = {'data': df, 'x': x_col, 'ax': ax}
                if hue_col:
                    ecdf_params['hue'] = hue_col
                sns.ecdfplot(**ecdf_params)
            
            elif plot_key == 'heatmap':
                plt.close()
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap=palette, ax=ax, fmt='.2f', 
                           linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
            
            elif plot_key == 'pairplot':
                plt.close()
                cols = extra_options.get('pair_cols', numeric_cols[:4])
                if cols:
                    pair_data = df[cols].dropna()
                    g = sns.pairplot(pair_data, diag_kind='kde', corner=True)
                    fig = g.fig
                else:
                    st.warning("Please select columns for pair plot")
                    fig, ax = plt.subplots()
            
            elif plot_key == 'jointplot':
                plt.close()
                joint_params = {'data': df, 'x': x_col, 'y': y_col, 'kind': 'scatter'}
                g = sns.jointplot(**joint_params)
                fig = g.fig
            
            elif plot_key == 'regplot':
                sns.regplot(data=df, x=x_col, y=y_col, ax=ax, scatter_kws={'alpha': 0.6})
            
            elif plot_key == 'scatter_matrix':
                plt.close()
                cols = extra_options.get('pair_cols', numeric_cols[:4])
                if cols:
                    pd.plotting.scatter_matrix(df[cols], figsize=(fig_width, fig_height), 
                                               diagonal='kde', alpha=0.7)
                    fig = plt.gcf()
            
            elif plot_key == 'distplot':
                sns.histplot(data=df, x=x_col, kde=True, ax=ax)
            
            elif plot_key == 'rugplot':
                sns.rugplot(data=df, x=x_col, ax=ax, height=0.5)
                sns.kdeplot(data=df, x=x_col, ax=ax, fill=True, alpha=0.3)
            
            elif plot_key == 'stripplot':
                strip_params = {'data': df, 'x': x_col, 'ax': ax}
                if y_col:
                    strip_params['y'] = y_col
                if hue_col:
                    strip_params['hue'] = hue_col
                sns.stripplot(**strip_params)
                plt.xticks(rotation=45, ha='right')
            
            elif plot_key == 'swarmplot':
                swarm_params = {'data': df, 'x': x_col, 'ax': ax}
                if y_col:
                    swarm_params['y'] = y_col
                if hue_col:
                    swarm_params['hue'] = hue_col
                # Limit data for swarmplot to avoid performance issues
                if len(df) > 500:
                    sample_df = df.sample(500)
                    swarm_params['data'] = sample_df
                    st.info("📌 Showing sample of 500 points for better performance")
                sns.swarmplot(**swarm_params)
                plt.xticks(rotation=45, ha='right')
            
            elif plot_key == 'countplot':
                count_params = {'data': df, 'x': x_col, 'ax': ax}
                if hue_col:
                    count_params['hue'] = hue_col
                # Limit categories if too many
                if len(df[x_col].unique()) > 20:
                    top_cats = df[x_col].value_counts().head(20).index
                    count_params['data'] = df[df[x_col].isin(top_cats)]
                    st.info("📌 Showing top 20 categories")
                sns.countplot(**count_params)
                plt.xticks(rotation=45, ha='right')
            
            elif plot_key == 'catplot':
                plt.close()
                cat_params = {'data': df, 'x': x_col, 'y': y_col if y_col else None, 'kind': 'bar'}
                if hue_col:
                    cat_params['hue'] = hue_col
                g = sns.catplot(**cat_params)
                fig = g.fig
            
            elif plot_key == 'pointplot':
                point_params = {'data': df, 'x': x_col, 'y': y_col, 'ax': ax}
                if hue_col:
                    point_params['hue'] = hue_col
                sns.pointplot(**point_params)
                plt.xticks(rotation=45, ha='right')
            
            elif plot_key == 'barplot_sns':
                bar_params = {'data': df, 'x': x_col, 'y': y_col, 'ax': ax}
                if hue_col:
                    bar_params['hue'] = hue_col
                if len(df[x_col].unique()) > 20:
                    top_cats = df[x_col].value_counts().head(20).index
                    bar_params['data'] = df[df[x_col].isin(top_cats)]
                    st.info("📌 Showing top 20 categories")
                sns.barplot(**bar_params)
                plt.xticks(rotation=45, ha='right')
            
            # Set labels and title
            if hasattr(ax, 'set_title') and plot_key not in ['pairplot', 'jointplot', 'catplot', 'scatter_matrix']:
                ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=20)
                if x_label:
                    ax.set_xlabel(x_label, fontsize=12)
                if y_label:
                    ax.set_ylabel(y_label, fontsize=12)
            
            plt.tight_layout()
            
            # Display plot
            st.pyplot(fig)
            
            # Download section
            st.markdown("---")
            st.markdown('<p class="section-header">💾 Download Your Plot</p>', unsafe_allow_html=True)
            
            download_col1, download_col2, download_col3, download_col4 = st.columns(4)
            
            # Save plot to different formats
            def get_plot_download(fig, format_type):
                buf = BytesIO()
                fig.savefig(buf, format=format_type, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                buf.seek(0)
                return buf
            
            with download_col1:
                png_buf = get_plot_download(fig, 'png')
                st.download_button(
                    label="📥 Download PNG",
                    data=png_buf,
                    file_name=f"{plot_title.replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with download_col2:
                pdf_buf = get_plot_download(fig, 'pdf')
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_buf,
                    file_name=f"{plot_title.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with download_col3:
                svg_buf = get_plot_download(fig, 'svg')
                st.download_button(
                    label="📥 Download SVG",
                    data=svg_buf,
                    file_name=f"{plot_title.replace(' ', '_')}.svg",
                    mime="image/svg+xml",
                    use_container_width=True
                )
            
            with download_col4:
                jpg_buf = get_plot_download(fig, 'jpg')
                st.download_button(
                    label="📥 Download JPG",
                    data=jpg_buf,
                    file_name=f"{plot_title.replace(' ', '_')}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            
            plt.close()
            
        except Exception as e:
            st.error(f"❌ Error generating plot: {str(e)}")
            st.info("💡 Tip: Make sure you've selected appropriate columns for this plot type.")

else:
    # Welcome screen when no data is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>👋 Welcome to Data Visualization Studio!</h2>
        <p style="font-size: 1.2rem; color: #666; margin: 2rem 0;">
            Upload your dataset using the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="custom-card" style="text-align: center; height: 200px;">
            <h3>📁</h3>
            <h4>Easy Upload</h4>
            <p>Support for CSV and Excel files. Simply drag and drop your data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card" style="text-align: center; height: 200px;">
            <h3>🎨</h3>
            <h4>20+ Plot Types</h4>
            <p>From basic charts to advanced statistical visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="custom-card" style="text-align: center; height: 200px;">
            <h3>💾</h3>
            <h4>Export Options</h4>
            <p>Download in PNG, PDF, SVG, or JPG formats.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Supported plot types
    st.markdown("---")
    st.markdown("### 📊 Supported Plot Types")
    
    plot_col1, plot_col2, plot_col3, plot_col4, plot_col5 = st.columns(5)
    
    with plot_col1:
        st.markdown("""
        **📊 Basic Plots**
        - Line Plot
        - Bar Chart
        - Scatter Plot
        - Area Chart
        - Pie Chart
        """)
    
    with plot_col2:
        st.markdown("""
        **📈 Statistical**
        - Histogram
        - Box Plot
        - Violin Plot
        - KDE Plot
        - ECDF Plot
        """)
    
    with plot_col3:
        st.markdown("""
        **🔗 Relationships**
        - Heatmap
        - Pair Plot
        - Joint Plot
        - Regression Plot
        - Scatter Matrix
        """)
    
    with plot_col4:
        st.markdown("""
        **📉 Distribution**
        - Dist Plot
        - Rug Plot
        - Strip Plot
        - Swarm Plot
        """)
    
    with plot_col5:
        st.markdown("""
        **🎯 Categorical**
        - Count Plot
        - Cat Plot
        - Point Plot
        - Bar Plot
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>Made with ❤️ using Streamlit, Matplotlib & Seaborn</p>
</div>
""", unsafe_allow_html=True)
