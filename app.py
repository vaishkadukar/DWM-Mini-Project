import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

#Horizontal menu
selector = option_menu(menu_title='Menu Options',options=['Cleaning','Visualisations','Classification'],default_index=1,orientation='horizontal')
with st.container():
    st.write('hello from header')

if selector == "Visualisations":
    with st.container():
        st.title("Choose a File: ")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            df = pd.read_csv(uploaded_file) # Use 'openpyxl' as the engine for xlsx files :engine='openpyxl'
            st.subheader("Data from Excel File:")
            st.dataframe(df)
            # print(df)

            selected_columns = st.multiselect("Select columns for visualization", df.columns)

            if not selected_columns:
                st.warning("Please select at least one column for visualization.")
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
