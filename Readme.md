# Project Setup & Run Guide

This guide explains how to set up and run both the **backend (Python API)** and **frontend (React app)** for this project.

---

## 📁 Project Structure

You should be working from the root folder:

```text
project-root/
│
├── backend/
├── frontend/
└── venv/
```

> ⚠️ Important: The virtual environment must be created in `project-root`, NOT inside the backend folder.

---

# 🧩 Backend Setup (Python)

### ✅ Step 1: Navigate to project root

Open PowerShell or terminal and run:

```bash
cd project-root
```

---

### ✅ Step 2: Create virtual environment

```bash
python -m venv venv
```

This will create:

```text
project-root/venv/
```

---

### ✅ Step 3: Activate virtual environment

```bash
venv\Scripts\activate
```

You should now see this in your terminal:

```text
(venv)
```

---

### ✅ Step 4: Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

---

### ✅ Step 5: Run the backend server

```bash
python -m uvicorn api:app --reload
```

Your backend API should now be running successfully 🚀

---

# 🎨 Frontend Setup (React)

Your frontend is located at:

```text
project-root/frontend/
```

### ✅ Step 1: Navigate to frontend

```bash
cd project-root/frontend
```

---

### ✅ Step 2: Install dependencies (ONLY first time)

```bash
npm install
```

This will create the `node_modules` folder.

---

### ✅ Step 3: Start React development server

```bash
npm start
```

Your React app will open in the browser at:

```text
http://localhost:3000
```

---

## ✅ Quick Command Summary

| Task                  | Command                              |
| --------------------- | ------------------------------------ |
| Create venv           | `python -m venv venv`                |
| Activate venv         | `venv\\Scripts\\activate`            |
| Install backend deps  | `pip install -r requirements.txt`    |
| Run backend           | `python -m uvicorn api:app --reload` |
| Install frontend deps | `npm install`                        |
| Run frontend          | `npm start`                          |

---

## 🛠 Troubleshooting

* Ensure you are in the correct directory before running commands
* Always activate the virtual environment before installing Python packages
* If `uvicorn` is not found, run:

  ```bash
  pip install uvicorn
  ```
* If frontend fails to start, delete `node_modules` and run:

  ```bash
  npm install
  ```

---

✅ You are now ready to develop and run the project locally. Happy coding! 🎉
