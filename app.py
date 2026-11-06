import streamlit as st
import io
import sys
import matplotlib.pyplot as plt

# Force matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Import your original functions from final_project
import final_project

st.set_page_config(page_title="Kyber HOCPA Simulation", layout="wide")

st.title("🛡️ Bounded Secure Adaptive Reduction Simulation on Masked Kyber")
st.write("This interactive application runs a Higher-Order Correlation Power Analysis (HOCPA) framework to evaluate side-channel vulnerabilities.")

if st.button("🚀 Execute Simulation Framework", type="primary"):
    st.subheader("📋 Execution Console Logs")
    
    # Visual loading spinner while computations run
    with st.spinner("Running calculations... please wait (takes around 30-60 seconds)"):
        # Capture terminal output prints to display them on the webpage
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            # Run your main experiment loop
            final_project.main()
            
            # Put terminal text back to normal
            sys.stdout = old_stdout
            console_output = buffer.getvalue()
            
            # Display text logs cleanly on screen
            st.code(console_output, language="text")
            
            # Display the generated plot automatically
            st.subheader("📊 Comparative Evaluation Graphs")
            st.pyplot(plt.gcf())
            
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"An error occurred: {e}")

