\# Privacy-First Image Sanitizer (TikTok TechJam 2025 – Track 7)



\*\*Protect your privacy instantly.\*\*  

This tool removes EXIF metadata and automatically detects \& masks \*\*faces\*\* and \*\*QR codes\*\* from images.  

You can choose between \*\*blur\*\* and \*\*pixelate\*\*, with optional \*\*YOLOv8 face detection\*\* for small/child faces.



---



\## ✨ Features

\- ✅ Strip EXIF data (GPS, device info, etc.)

\- ✅ Auto-detect faces (OpenCV Haar)  

\- ✅ Auto-detect QR codes (OpenCV QRDetector)  

\- ✅ Manual bounding box input (x,y,w,h)  

\- ✅ Blur / Pixelate masking options  

\- ✅ Optional YOLOv8 face detection (local only – needs weights)  

\- ✅ Streamlit UI or CLI mode  



---



## 🚀 Demo
- **Streamlit Cloud (UI)** – [🔗 Live Demo](https://privacy-sanitizer.streamlit.app) *(YOLO disabled on cloud)*  
- **GitHub Repo** – [🔗 Source Code](https://github.com/anshikaaaa17/privacy-sanitizer)  
- **Demo Video** – [🔗 YouTube](https://youtu.be/your-demo-video)  




---





\## 📂 Project Structure

.

├── app.py # Main application (UI + CLI)

├── requirements.txt # Python dependencies

├── weights/ # YOLOv8 face weights (local only, not on GitHub)

└── README.md # This file

