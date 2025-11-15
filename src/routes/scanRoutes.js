import express from "express";
import multer from "multer";
import { storage } from "../config/cloudinary.js";
import { createScan, listScans, getStats } from "../controllers/scanController.js";

const router = express.Router();
const upload = multer({ storage });

// 🟢 Route for uploading CTG image
router.post("/", upload.single("ctgImage"), createScan);

// 🟡 Route for listing all scans
router.get("/scans", listScans);

// 🔵 Route for stats
router.get("/scans/stats", getStats);

export default router;
