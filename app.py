import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image

favicon = Image.open("data-mining.png")
st.set_page_config(page_title='DWM Mini Project', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')


#Horizontal menu
selector = option_menu(menu_title='Welcome!', options=['Home','Cleaning Data','Visualizing Data','Exploring Data','Clustering Data'], default_index=0, orientation='horizontal', icons=['home','v','v','v','v'])

if "data" not in st.session_state:
    st.session_state["data"] = ""

if selector == "Home":
    with st.container():
        st.subheader("Choose a File: ")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            df = pd.read_csv(uploaded_file) # Use 'openpyxl' as the engine for xlsx files
            st.text("Data from CSV File:")
            st.dataframe(df)
            st.session_state['data']= df
            

if selector == "Visualizing Data":
    with st.container():
        df = st.session_state['data']
        selected_columns = st.multiselect("Select columns for visualization", df.columns) 

        if len(selected_columns) < 2:
            st.warning("Please select at least two columns for visualization.")
            
        else:
            # Choose the type of visualization
            plot_type = st.selectbox("Select the type of visualization", ["Bar Chart", "Line Chart","Scatter Plot", "Box Plot","Histogram","Pie Chart", "Heatmap", "Area Chart", "Violin Plot"])

            # Create the selected plot
            st.subheader(f"{plot_type} based on selected columns:")
            if plot_type == "Bar Chart":
                fig = px.bar(df, x=selected_columns[0] ,y=selected_columns[1], title='Bar Chart', color_discrete_sequence=["#8785A2"]*len(selected_columns))
                st.plotly_chart(fig)
            elif plot_type == "Line Chart":
                fig = px.line(df, x=selected_columns,y=selected_columns[1], title='Line Chart')
                st.plotly_chart(fig)
            elif plot_type == "Scatter Plot":
                fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1], title='Scatter Plot')
                st.plotly_chart(fig)
            elif plot_type == "Box Plot":
                fig, ax = plt.subplots()
                sns.boxplot(x=selected_columns[0], y=selected_columns[1], data=df, ax=ax)
                st.pyplot(fig)
            elif plot_type == "Histogram":
                fig = px.histogram(df, x=selected_columns[0], title='Histogram')
                st.plotly_chart(fig)
            elif plot_type == "Pie Chart":
                fig = px.pie(df, names=selected_columns[0], title='Pie Chart')
                st.plotly_chart(fig)
            elif plot_type == "Heatmap":
                if selected_columns:
                    # Select only the relevant columns from the DataFrame
                    df_subset = df[selected_columns]

                    # Check if the selected columns are numeric
                    if df_subset.select_dtypes(include='number').shape[1] == len(selected_columns):
                        # Create a heatmap using Plotly Express
                        fig = px.imshow(df_subset.corr(), x=selected_columns, y=selected_columns, title='Heatmap')
                        st.plotly_chart(fig)
                    else:
                        st.error("Selected columns must contain numeric data for the heatmap.")
                else:
                    st.warning("Please select columns for the heatmap.")
            elif plot_type == "Area Chart":
                fig = px.area(df, x=selected_columns[0], y=selected_columns[1], title='Area Chart')
                st.plotly_chart(fig)
            elif plot_type == "Violin Plot":
                fig = px.violin(df, x=selected_columns[0], y=selected_columns[1], title='Violin Plot')
                st.plotly_chart(fig)

if selector == "Exploring Data":
    df = st.session_state['data']

    numeric_columns = df.select_dtypes(include=['number']).columns
    selected_column= st.selectbox("Select a column for exploration", numeric_columns)
    column_data = df[selected_column]
    if not selected_column:
        st.warning("Please select one column for visualization.")
    else:
        mean_value = column_data.mean()
        median_value = column_data.median()
        # mode_value = mode(column_data).mode[0]
        max_value = column_data.max()
        min_value = column_data.min()
        midrange_value = (max_value + min_value) / 2
        data_range = max_value - min_value
        quartiles = column_data.quantile([0.25, 0.5, 0.75])
        iqr_value = quartiles[0.75] - quartiles[0.25]
        variance_value = column_data.var()
        std_deviation_value = column_data.std()

        st.subheader(f"Statistics for {selected_column}:")
        st.write(f"Mean: {mean_value}")
        st.write(f"Median: {median_value}")
        # st.write(f"Mode: {mode_value}")
        st.write(f"Maximum: {max_value}")
        st.write(f"Minimum: {min_value}")
        st.write(f"Midrange: {midrange_value}")
        st.write(f"Range: {data_range}")
        st.write(f"Quartiles: Q1={quartiles[0.25]}, Q2={quartiles[0.5]}, Q3={quartiles[0.75]}")
        st.write(f"IQR (Interquartile Range): {iqr_value}")
        st.write(f"Variance: {variance_value}")
        st.write(f"Standard Deviation: {std_deviation_value}")

        st.set_option('deprecation.showPyplotGlobalUse', False)
        # st.subheader(f"Histogram for {selected_column}:")
        # plt.hist(column_data, bins=20, color='skyblue', edgecolor='black')
        # st.pyplot()

        plt.hist(column_data, bins='auto', density=True, alpha=0.7, color='skyblue',edgecolor='black',label='Histogram')
        x = range(int(min(column_data)), int(max(column_data)) + 1)
        pdf = norm.pdf(x, mean_value, std_deviation_value)
        plt.plot(x, pdf, color='red', label='Frequency Curve')
        plt.legend()
        plt.show()
        st.pyplot()

if selector == "Cleaning Data":
    st.write("Cleaning")


if selector == "Clustering Data":
    st.write("Clustering")
    df = st.session_state['data']
    selected_columns = st.multiselect("Select columns for visualization", df.columns) 
    num_clusters = st.number_input("Enter the number of clusters", min_value=2, max_value=10, value=3, step=1)

    if len(selected_columns) < 2:
        st.warning("Please select at least two columns for visualization.")
    else:
        temp = df
        data_for_clustering = temp[selected_columns]
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_for_clustering)
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)
        # df['Cluster'] = clusters
        temp['Cluster'] = clusters
        st.subheader("Scatter Plot of Clusters:")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=selected_columns[0], y=selected_columns[1], hue='Cluster', palette='viridis', ax=ax)
        st.pyplot(fig)