#trial code..not the main file
import streamlit as st
from streamlit_option_menu import option_menu

selected = option_menu(
    menu_title="Menu",
    options=["Home", "Visualizations","Classification"]
    orientation="horizontal"
)

selector = option_menu(menu_title='Menu Options',options=['Cleaning','Visualisations',"Classification"],default_index=1,orientation='horizontal')