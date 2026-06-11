# Image Labeling Guide — Surgical Instrument Detection
### Instructions for the Annotation Team

---

## Overview

You will be drawing bounding boxes around surgical instruments in a set of **126 images**.  
Each box must be assigned the correct instrument name from a fixed list of 56 classes.

The tool used is **Label Studio** — free, runs entirely on your own computer, no internet required once installed.

Estimated effort: approximately **3–5 minutes per image** depending on how many instruments appear.

---

## Section 1 — What You Need

- A computer running **Windows 10/11**, **macOS**, or **Ubuntu Linux**
- **Python 3.8 or higher** installed
  - Check by opening a terminal and typing: `python --version`
  - Download from https://www.python.org/downloads/ if missing
- The folder of images to label (sent as a zip file — unzip it before starting)
- At least **4 GB free disk space**

---

## Section 2 — Installation (do this once)

Open a terminal (Command Prompt or PowerShell on Windows, Terminal on Mac/Linux) and run:

```
pip install label-studio
```

Wait for it to finish. Then start Label Studio:

```
label-studio start
```

Your browser will open automatically at `http://localhost:8080`.  
If it does not open, type that address into your browser manually.

**Create an account** on the first-run screen (this is local only — no data is sent anywhere).

---

## Section 3 — Create the Project

1. Click **Create Project** (top-right).
2. Enter the project name: `Surgical Instruments`
3. Click the **Labeling Setup** tab at the top of the dialog.
4. Select **Object Detection with Bounding Boxes**.
5. You will see an XML code box. **Delete everything in it** and paste the entire block below:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Acromioplasty Retractor"/>
    <Label value="Agrawal Talon Retractor"/>
    <Label value="Angled Glenoid Retractor"/>
    <Label value="Angled Glenoid Retractor Forked"/>
    <Label value="Anterior Glenoid Neck Retractor"/>
    <Label value="Bacastow Arthroscopic Deltoid Lift Retractor"/>
    <Label value="Bacastow Axillary Nerve Retractor with Suction"/>
    <Label value="Bacastow Glenoid Retractor"/>
    <Label value="Bell-Hawkins Shoulder Frame and Blade Set"/>
    <Label value="Blount Retractor with Small Handle"/>
    <Label value="Bolanos Modified Chandler Retractor"/>
    <Label value="Browne-Deltoid Retractor"/>
    <Label value="Burkhead Glenoid Retractor"/>
    <Label value="Burkhead Reversible TSARSA Retractor"/>
    <Label value="Capsule Retractors"/>
    <Label value="Chandler Retractor"/>
    <Label value="Chandran Distal Biceps Tissue Protector"/>
    <Label value="Deltoid Retractor"/>
    <Label value="Dillon Curved Glenoid Retractors"/>
    <Label value="Dingo Modified Humeral Head Retractor"/>
    <Label value="Evans Modified Fukuda-type Retractors"/>
    <Label value="George Semi-Circumferential Glenoid Retractor"/>
    <Label value="Glenoid Access Retractor"/>
    <Label value="Glenoid Neck Retractor"/>
    <Label value="Glenosphere Component Retractor"/>
    <Label value="Goldstein Glenoid Neck Retractor"/>
    <Label value="Gunther Glenoid Retractor"/>
    <Label value="Horseshoe Shoulder Frame and Blade Assembly"/>
    <Label value="Humeral Head Retractor"/>
    <Label value="Incavo Reverse Cutting Osteotomes"/>
    <Label value="Kaminsky OrthoLucent Browne-type Deltoid Retractors"/>
    <Label value="Kirschenbaum Acromioplasty Retractor"/>
    <Label value="Large Deltoid Retractor"/>
    <Label value="Latarjet Retractor"/>
    <Label value="Levy Anterior Glenoid Retractor"/>
    <Label value="Levy Wide Deltoid Retractor"/>
    <Label value="Locking Bone Screw Assembly Set"/>
    <Label value="Long Handled Fukuda Type Retractors"/>
    <Label value="McFarland Bent Cobb Elevator"/>
    <Label value="McFarland Shoulder V Retractor"/>
    <Label value="Mehalik Posterior Glenoid Retractor with Long Handle"/>
    <Label value="Modified Darrach-type Bent Elevator"/>
    <Label value="Modified Fukuda-type Retractor with Reamer Slot"/>
    <Label value="Modified Humeral Head Retractors"/>
    <Label value="Modified Winged Fukuda Retractor"/>
    <Label value="OrthoLucent Modified Fukuda-type Retractor"/>
    <Label value="Posterior Glenoid Neck Retractor"/>
    <Label value="Rogozinski Glenoid Reaming Retractor"/>
    <Label value="Rogozinski Glenoid Retractor"/>
    <Label value="Rogozinski Reverse Angle Retractor"/>
    <Label value="Superior Coracoid Retractor"/>
    <Label value="Vaughan Distal Bicep Tendon Repair Retractor"/>
    <Label value="Weatherly Mini-Deltoid Retractors"/>
    <Label value="Wiater Shoulder Bone Hook"/>
    <Label value="Woods Retractor"/>
    <Label value="Unknown Surgical Instrument"/>
  </RectangleLabels>
</View>
```

6. Click **Save** at the bottom of the dialog.

---

## Section 4 — Upload the Images

1. Inside the project, click **Import** (top-right area).
2. Drag and drop all 126 image files from the folder provided, or click **Upload Files** to browse.
3. Click **Import** to confirm.
4. You should see all images listed as tasks with status **Unlabeled**.

---

## Section 5 — How to Label an Image

Click **Label All Tasks** to begin. Label Studio will show one image at a time.

### The labeling screen has three parts:
- **Centre** — the image
- **Right panel** — list of class labels
- **Bottom bar** — submit / skip controls

---

### Step-by-step for each image:

**1. Read the filename** — it tells you what instrument is shown.  
For example: `Burkhead_Glenoid_Retractor_24.jpg` → use class **Burkhead Glenoid Retractor**.

**2. Select a class** from the right panel by clicking it.  
The selected class is highlighted. You can also type in the search box at the top of the label list to find a class quickly.

**3. Draw a bounding box.**  
Click and drag a rectangle tightly around the instrument. Include the entire tool — handle to tip — but exclude background as much as possible.

**4. If there are multiple instruments in one image**, repeat steps 2–3 for each one. Each instrument gets its own box.

**5. Click Submit** to save and move to the next image.

---

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `R` | Rectangle (bounding box) tool |
| `Ctrl + Z` | Undo last box |
| `Ctrl + Backspace` | Delete selected box |
| `Tab` | Select next annotation |

---

## Section 6 — Labeling Rules

### DO:
- Draw boxes as **tight as possible** around the instrument
- Label **every** instrument visible in the image, including partial ones at the edge of frame
- Use the filename as your primary guide to the correct class name
- Zoom in (scroll wheel) for small or cluttered images to be precise

### DO NOT:
- Do not leave any image without at least one bounding box
- Do not draw one large box around the entire image
- Do not include background, text labels, or ruler scales inside the box

### When to use **Unknown Surgical Instrument**:
Use this class when:
- The filename contains `Medical_Device_No_specific_alt_text_provided`
- The filename contains `Figure_`, `Part_Number`, `Implants`, or similar non-specific names
- You genuinely cannot identify what the instrument is

---

## Section 7 — Saving Your Progress

Label Studio saves automatically every time you click **Submit**.  
You can close the browser and reopen it at `http://localhost:8080` at any time — your work will be there.

To resume: open the project → click **Label All Tasks** → it continues from where you left off.

---

## Section 8 — Exporting the Finished Labels

Once all 126 images show status **Labeled** (check the project dashboard — 0 Unlabeled):

1. Go to the project main page.
2. Click the **Export** button (top-right).
3. Select **YOLO** from the format list.
4. Click **Export** — this downloads a `.zip` file.

**That zip file is what you send back to us. Do not unzip it or modify its contents.**

Name the file before sending:
```
surgical_labels_YYYYMMDD.zip
```
(replace YYYYMMDD with today's date, e.g. `surgical_labels_20260615.zip`)

---

## Section 9 — Quality Checklist Before Sending

Go through this before exporting:

- [ ] All 126 images have been labeled (project dashboard shows 0 Unlabeled)
- [ ] Every visible instrument has a bounding box
- [ ] No box is left without a class assigned
- [ ] The exported zip opens and contains an `images/` and `labels/` folder

---

## Section 10 — Common Questions

**Q: The filename matches a class name exactly — do I still need to check the image?**  
A: Yes. Always look at the image. Some images show more than one instrument.

**Q: I can't find the right class name in the list.**  
A: Use **Unknown Surgical Instrument**. Do not leave the image without any annotation.

**Q: The image is very small or blurry. Should I still label it?**  
A: Yes. Draw the best box you can and note the filename in your delivery email.

**Q: The image shows a line drawing or diagram, not a photograph.**  
A: Label it the same way. Draw the box around the drawn instrument.

**Q: Label Studio is running slowly.**  
A: Close other browser tabs. If still slow, restart Label Studio (`Ctrl+C` in the terminal, then `label-studio start` again) — your work is saved automatically.

**Q: I submitted an image with a wrong label.**  
A: Go back to it from the task list, click the existing annotation, delete the wrong box, and redraw it correctly. Click **Update** to save the correction.

---

## Contact

If you have any questions about a specific image or instrument, contact us before guessing.  
It is better to ask than to mislabel.
