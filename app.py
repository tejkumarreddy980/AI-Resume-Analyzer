from flask import Flask,render_template,request
import pdfplumber
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def extract_text(file):
    if file.filename.endswith(".pdf"):
        text=""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text+=page.extract_text()
        return text

    elif file.filename.endswith(".docx"):
        doc=docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    return ""

@app.route("/",methods=["GET","POST"])
def index():
    score=None
    if request.method=="POST":
        resume=request.files["resume"]
        jd=request.form["jd"]

        resume_text=extract_text(resume)

        vectorizer=TfidfVectorizer()
        vectors=vectorizer.fit_transform([resume_text,jd])
        score=cosine_similarity(vectors[0],vectors[1])[0][0]
        score=round(score*100,2)

    return render_template("index.html",score=score)

if __name__=="__main__":
    app.run(debug=True)
