import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk
from tkinter import font as tkfont

# --- 1. ML BACKEND (Optimized) ---
print("Loading Model...")
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = "".join([char for char in text if char.isalpha() or char.isspace()])
    words = text.lower().split()
    return " ".join([w for w in words if w not in stop_words])

df['text'] = df['text'].apply(clean_text)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# --- 2. THE MODERN UI ---
def check_spam():
    user_input = text_entry.get("1.0", "end-1c")
    if not user_input.strip(): return
    
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    
    if prediction == 1:
        result_label.config(text="⚠️ SPAM DETECTED", fg="#E74C3C")
        status_box.config(bg="#FDEDEC")
    else:
        result_label.config(text="✅ LEGITIMATE (HAM)", fg="#27AE60")
        status_box.config(bg="#EAFAF1")

# Main Window Setup
root = tk.Tk()
root.title("AI Spam Classifier")
root.geometry("500x550")
root.configure(bg="#2C3E50") # Dark professional background

# Custom Fonts
title_font = tkfont.Font(family="Helvetica", size=20, weight="bold")
norm_font = tkfont.Font(family="Helvetica", size=11)

# Main Container
main_frame = tk.Frame(root, bg="#ECF0F1", bd=0, padx=30, pady=30)
main_frame.place(relx=0.5, rely=0.5, anchor="center", width=440, height=480)

# Title
tk.Label(main_frame, text="Spam Sentinel", font=title_font, fg="#2C3E50", bg="#ECF0F1").pack(pady=(0, 10))
tk.Label(main_frame, text="Naïve Bayes Machine Learning Model", font=("Helvetica", 9, "italic"), fg="#7F8C8D", bg="#ECF0F1").pack()

# Input Area
tk.Label(main_frame, text="Paste Email Content Below:", font=norm_font, fg="#34495E", bg="#ECF0F1").pack(pady=(20, 5), anchor="w")
text_entry = tk.Text(main_frame, height=8, font=("Courier", 10), bd=1, relief="solid", highlightthickness=1, highlightcolor="#3498DB")
text_entry.pack(fill="x")

# Analyze Button
btn_check = tk.Button(main_frame, text="ANALYZE MESSAGE", command=check_spam, bg="#3498DB", fg="white", font=("Helvetica", 11, "bold"), bd=0, cursor="hand2", activebackground="#2980B9", pady=10)
btn_check.pack(fill="x", pady=20)

# Result Area (The "Card")
status_box = tk.Frame(main_frame, bg="#D5DBDB", pady=15, bd=1, relief="flat")
status_box.pack(fill="x")

result_label = tk.Label(status_box, text="READY TO SCAN", font=("Helvetica", 14, "bold"), bg=status_box.cget("bg"), fg="#7F8C8D")
result_label.pack()

root.mainloop()