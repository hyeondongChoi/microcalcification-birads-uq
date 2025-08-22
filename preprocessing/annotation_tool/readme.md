## Microcalcification Annotation Tool

PyQt5-based tool for annotating **microcalcification regions** in mammography images.  
Each image supports **up to 3 bounding-box annotations**.  
To match corresponding microcalcifications between the **CC** and **MLO** views of the same breast, annotations are **color-coded consistently across views**.  

## Shortcuts
- **Zoom in**: `Ctrl + Wheel Up`  
- **Zoom out**: `Ctrl + Wheel Down`  
- **Save annotations**: `Ctrl + S`  
- **Remove bounding box**: `Right Click`

## Build Executable (Optional)
You can build an executable with PyInstaller 6.10.0:

```bash
pyinstaller annotation_tool.py
```
