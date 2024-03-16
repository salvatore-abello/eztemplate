from flask import Flask, request
from eztemplate import render_template

app = Flask(__name__)

@app.get("/")
def index():
    user = request.args.get("user", "guest")
    
    return render_template("index.html", user=user)

if __name__ == "__main__":
    app.run("0.0.0.0", 1337)