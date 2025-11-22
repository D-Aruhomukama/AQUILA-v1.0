import streamlit as st
import numpy as np
import pickle
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

svc = joblib.load("shv_svc.pkl")
vectorizer = joblib.load("kmer_vectorizer.joblib")

#loaded_model = pickle.load(open('blashv_model.pkl', 'rb'))
st.sidebar.write("   ")
st.sidebar.write("   ")
st.sidebar.write("   ")
st.sidebar.image("image2.jpeg")

st.write("""
# blaSHV Gene Prediction

blaSHV is 1 of 3 commonest ABR genes, that confers resistance to Cephems/3rd-Generation 
Cephalosporins particularly Ceftazidime.
""")
label_map = {
    1: "Resistant",
    0: "Non-resistant"
}

# Convert sequence into k-mers
def get_kmers(seq, k=6):
    return " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])

def read_fasta_merge_groups(file):
    """
    Reads a FASTA file and merges sequence fragments that share the same
    numeric prefix before the underscore.

    Example:
       >5_1, >5_2, >5_3  → combined under key "5"
       >10_1, >10_2      → combined under key "10"
    """

    sequences = {}
    current_group = None
    current_seq = []

    for raw in file:
        line = raw.decode("utf-8").strip()
        if not line:
            continue

        # Header line
        if line.startswith(">"):
            identifier = line[1:].strip()

            # Extract group before underscore:
            # "5_1" -> "5", "10_3" -> "10"
            group = identifier.split("_")[0]

            # Save accumulated sequence for the previous entry
            if current_group:
                continuous = "".join(current_seq)
                if current_group in sequences:
                    sequences[current_group] += continuous
                else:
                    sequences[current_group] = continuous

            # Start new entry
            current_group = group
            current_seq = []

        else:
            # Sequence line
            sequence_line = line.replace(" ", "").upper()
            current_seq.append(sequence_line)

    # Save LAST sequence
    if current_group:
        continuous = "".join(current_seq)
        if current_group in sequences:
            sequences[current_group] += continuous
        else:
            sequences[current_group] = continuous

    return sequences


# Streamlit UI
# -------------- OPTION 1: FASTA FILE ----------------
fasta_file = st.file_uploader("Upload FASTA File", type=["fasta", "fas","fa", "txt"])

pdf_data = None
if fasta_file:
    st.info("FASTA file uploaded successfully.")

    sequences = read_fasta_merge_groups(fasta_file)

    results = []
    for group_id, sequence in sequences.items():
        kmers = get_kmers(sequence)
        X = vectorizer.transform([kmers])
        raw_pred = svc.predict(X)[0]
        prediction = label_map[raw_pred]
        st.success(f"Predicted Label: **{prediction}**")

        results.append([group_id, sequence, raw_pred])

    st.subheader("Prediction Results")
    st.write("Predicted labels for each sequence:")

    import pandas as pd
    df_results = pd.DataFrame(results, columns=["ID", "Sequence", "Prediction"])
    st.dataframe(df_results)
    
    # ---- Create PDF in memory ----
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "DNA Sequence Classification Results")

    c.setFont("Helvetica", 12)
    y_position = 720

    for idx, row in df_results.iterrows():

        id_text = f"Sample ID: {row['ID']}"
        pred_text = f"Prediction: {label_map[row['Prediction']]}"

        c.drawString(50, y_position, id_text)
        y_position -= 20
        c.drawString(50, y_position, pred_text)
        y_position -= 20

        # ----- Add conclusion for Resistant -----
        if label_map[row['Prediction']] == "Resistant":
            conclusion_text = (
                "Conclusion: The isolate may be resistant to Cephems/3rd-Generation "
                "Cephalosporins, particularly Ceftriaxone and Ceftazidime. "
                "Report immediately to the Healthcare Provider. Perform additional "
                "testing for AmpC and Carbapenemase production to assess broader "
                "beta-lactam resistance."
            )

            text_obj = c.beginText(50, y_position)
            text_obj.setFont("Helvetica", 10)

            for line in conclusion_text.split(". "):
                text_obj.textLine(line.strip() + ".")
            c.drawText(text_obj)

            y_position -= 70
        else:
            y_position -= 20

        # New page if needed
        if y_position < 80:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = 750

    c.save()
    pdf_buffer.seek(0)

    # ---- Streamlit PDF Download Button ----
    st.download_button(
        label="Download Results as PDF",
        data=pdf_buffer,
        file_name="predictions.pdf",
        mime="application/pdf"
        )



    
# -------------- OPTION 2: MANUAL INPUT ---------------
st.subheader("Or Enter a Single DNA Sequence")
sequence_input = st.text_area("DNA Sequence", height=150)

if st.button("Predict Single Sequence"):
    if len(sequence_input.strip()) == 0:
        st.warning("Please enter a DNA sequence.")
    else:
        kmers = get_kmers(sequence_input.strip())
        X = vectorizer.transform([kmers])
        raw_pred = svc.predict(X)[0]
        prediction = label_map[raw_pred]
        st.success(f"Predicted Label: **{prediction}**")
        
        # ----- Create PDF for manual input -----
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 750, "DNA Sequence Classification Result")

        c.setFont("Helvetica", 12)
        c.drawString(50, 720, "Sample ID: Manual_Input")
        c.drawString(50, 700, f"Prediction: {prediction}")
        
        # ----- Add conclusion if Resistant -----
        if prediction == "Resistant":
            conclusion_text = (
            "Conclusion: The isolate may be resistant to Cephems/3rd-Generation "
            "Cephalosporins, particularly Ceftriaxone and Ceftazidime. "
            "Report immediately to the Healthcare Provider. Perform additional "
            "testing for AmpC and Carbapenemase production to assess broader "
            "beta-lactam resistance."
            )

        # Draw multi-line text
        text_obj = c.beginText(50, 660)
        text_obj.setFont("Helvetica", 11)

        for line in conclusion_text.split(". "):
            text_obj.textLine(line.strip() + ".")
        c.drawText(text_obj)

                
        c.save()
        pdf_buffer.seek(0)

        # ---- Streamlit Download Button ----
        st.download_button(
            label="Download Result as PDF",
            data=pdf_buffer,
            file_name="single_prediction.pdf",
            mime="application/pdf"
        )




