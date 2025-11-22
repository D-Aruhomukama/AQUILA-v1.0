import streamlit as st


st.set_page_config(
    page_title="Aquila",
    page_icon="🧬",
)

st.write("# Welcome to AQUILA! 🧬")

# st.sidebar.success("Select a page above.")
st.sidebar.write("   ")
st.sidebar.write("   ")
st.sidebar.write("   ")

st.sidebar.image("image2.jpeg")

st.markdown(
    """
        
    AQUILA was developed as part of “DEVELOPMENT OF A DETECTION PIPELINE, PREDICTION MODEL AND 
    USER INTERFACE FOR ESCHERICHIA COLI IN UGANDA”. The AQUILA workflow (AWF) enables ABR detection
    and prediction using sequence data for all levels of expertise. The pre-configured analysis packages are freely accessible from a laptop/desktop application with an easy-to-use graphical interface or the command line and can be run on local compute or in the cloud. AQUILA was specially developed for, but is not limited to, ESCHERICHIA COLI.
 """
)

st.markdown(
    """
    ### Developer
    Dickson Aruhomukama, with support from Hellen Nakabuye, Ronald Galiwango, and Benon Asiimwe; Makerere University Infectious Diseases Institute through the Professor Sewankambo Training Program for Global Health Security in Africa, and the Government of Uganda (MakRIF).
"""
)

st.markdown(
    """
    ### About Developer
    https://pubmed.ncbi.nlm.nih.gov/?term=DICKSON+ARUHOMUKAMA 
"""
)


