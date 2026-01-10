# 🛑 Shutting Down Public Deployment

Follow these steps to shut down your Vercel and Render deployments.

## 🔴 Shut Down Vercel (Frontend)

1. Go to: https://vercel.com/dashboard
2. Find your **`code-query-jade`** project
3. Click on the project
4. Go to **Settings** (top navigation)
5. Scroll to bottom → **Delete Project**
6. Type the project name to confirm
7. Click **Delete**

✅ Frontend is now shut down!

---

## 🔴 Shut Down Render (Backend)

1. Go to: https://dashboard.render.com
2. Find your **`codequery-backend`** service
3. Click on the service
4. Go to **Settings** (left sidebar)
5. Scroll to bottom → **Delete Web Service**
6. Type `DELETE` to confirm
7. Click **Delete Web Service**

✅ Backend is now shut down!

---

## 🧹 Optional: Disconnect from GitHub

If you want to prevent accidental redeployment:

### Vercel
1. Go to: https://vercel.com/dashboard
2. Click **Settings** (top right)
3. Go to **Git Integration**
4. Find CodeQuery repo → Click **Disconnect**

### Render
1. Already deleted with the service ✅

---

## ✅ Confirmation

- Your public URLs will no longer work:
  - ❌ `https://code-query-jade.vercel.app`
  - ❌ `https://codequery-backend.onrender.com`

- Your local setup will still work:
  - ✅ `http://localhost:5173` (frontend)
  - ✅ `http://localhost:8000` (backend)

---

## 💡 Note

After deletion:
- All deployment logs are removed
- Database data on Render is deleted
- No more charges from these services
- Your GitHub repo remains intact
- Local development continues as normal

---

**You can always redeploy later by pushing to GitHub (if integrations are still connected).**

