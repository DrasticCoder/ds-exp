import streamlit as st

st.set_page_config(page_title="Test", layout="wide")
st.title("Test App")
st.write("If you see this, the fix worked!")

tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
with tab1:
    st.write("Tab 1 content")
with tab2:
    st.write("Tab 2 content")
