import streamlit as st
from streamlit_option_menu import option_menu

selector = option_menu(menu_title='Menu Options',options=['Cleaning','Visualisations'],default_index=1,orientation='horizontal')
with st.container():
    st.write('hello from header')
    
