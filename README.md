# מערכת שאלות ותשובות - רשות המיסים

מערכת RAG (Retrieval-Augmented Generation) לעובדי מס הכנסה המתמחים בהחזרי מס.

## הגדרה ראשונית

### שלב 1 - קבלת מפתח Gemini API (חינם)
1. כנס ל: https://aistudio.google.com/app/apikey
2. לחץ "Create API Key"
3. שמור את המפתח

### שלב 2 - הרצת סקריפט ה-ingest (על המחשב המקומי שלך)

```bash
# התקן תלויות
pip install -r requirements.txt

# הגדר מפתח API
export GEMINI_API_KEY=your_key_here   # Linux/Mac
# או: set GEMINI_API_KEY=your_key_here   # Windows

# הרץ את הסקריפט עם הנתיב לתיקיית המסמכים שלך
python scripts/ingest.py --docs "C:/Users/YourName/אוריה- אפליקציות קלוד- קלוד קוד מס הכנסה"
```

הסקריפט יעשה:
- &#10003; יקרא את כל ה-PDF וקבצי Word מהתיקייה שלך
- &#10003; ישאב תוכן רלוונטי מאתר זכותי
- &#10003; יאסוף חוזרי ניתוב שלב א' מאתר רשות המיסים
- &#10003; ייצור קובץ `data/knowledge_base.json`

### שלב 3 - העלאה ל-Railway

```bash
git add data/knowledge_base.json
git commit -m "Add knowledge base"
git push origin claude/tax-knowledge-portal-fcaXi
```

### שלב 4 - הגדרת Railway

1. כנס ל-Railway Dashboard
2. צור פרויקט חדש מ-GitHub repo זה
3. הוסף משתנה סביבה: `GEMINI_API_KEY=your_key_here`
4. Railway יפרוס אוטומטית

## עדכון מאגר הידע

בכל פעם שיש מסמכים חדשים:
1. הרץ שוב: `python scripts/ingest.py --docs /path/to/docs`
2. Push לגיט - Railway מפרוס אוטומטית

## מבנה הפרויקט

```
tax-q-a/
├── app/
│   ├── main.py          # FastAPI server
│   └── rag.py           # RAG engine
├── scripts/
│   └── ingest.py        # Document ingestion + web scraping
├── static/
│   └── index.html       # Hebrew RTL frontend
├── data/
│   └── knowledge_base.json  # Generated - commit this!
└── railway.toml
```
