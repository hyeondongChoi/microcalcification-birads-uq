import os
import sys

from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout, 
                           QWidget, QPushButton, QHBoxLayout, QFileDialog, 
                           QListWidget, QSplitter)
from PyQt5.QtGui import QPixmap, QMouseEvent, QPainter
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtGui import QCursor

SCALE_FACTOR = 1.25
BBOX_SIZE = 256  # Set bounding box size

class ImageLabel(QLabel):
    bbox_status_changed = pyqtSignal(bool)  # Add signal

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.offset = QPoint(0, 0)
        self.dragging = False
        self.drag_start_position = QPoint()
        self.bbox_active = False
        self.bbox_positions = []  # Stored in original image coordinate system
        self.colors = [Qt.red, Qt.yellow, Qt.blue]
        self.zoom_factor = 1.0
        self.selected_bbox = None
        self.bbox_dragging = False
        self.bbox_drag_offset = None
        self.original_image_size = None  # Store original image size
        self.remove_active = False  # Variable to store the state of Remove mode activation
        
    def set_original_image_size(self, size):
        """Set the original image size"""
        self.original_image_size = size

    def viewer_to_original_coords(self, viewer_pos):
        """Convert viewer coordinates to original image coordinates"""
        if not self.original_image_size:
            return viewer_pos
            
        # Current displayed image size in the viewer
        current_width = self.pixmap().width()
        current_height = self.pixmap().height()
        
        # Calculate relative position in viewer (ratio between 0 and 1)
        relative_x = (viewer_pos.x() - self.offset.x()) / current_width
        relative_y = (viewer_pos.y() - self.offset.y()) / current_height
        
        # Map relative position to the original image size
        orig_x = int(relative_x * self.original_image_size.width())
        orig_y = int(relative_y * self.original_image_size.height())
        
        return QPoint(orig_x, orig_y)

    def original_to_viewer_coords(self, orig_pos):
        """Convert original image coordinates to viewer coordinates"""
        if not self.original_image_size:
            return orig_pos
            
        current_width = self.pixmap().width()
        current_height = self.pixmap().height()
        
        # Calculate relative position in original image (ratio between 0 and 1)
        relative_x = orig_pos.x() / self.original_image_size.width()
        relative_y = orig_pos.y() / self.original_image_size.height()
        
        # Map relative position to current viewer size
        viewer_x = int(relative_x * current_width + self.offset.x())
        viewer_y = int(relative_y * current_height + self.offset.y())
        
        return QPoint(viewer_x, viewer_y)

    def confine_to_image(self, pos):
        """Ensure bbox does not exceed image boundaries in the original image coordinate system"""
        if not self.original_image_size:
            return pos
            
        x = max(0, min(pos.x(), self.original_image_size.width() - BBOX_SIZE))
        y = max(0, min(pos.y(), self.original_image_size.height() - BBOX_SIZE))
        
        return QPoint(x, y)

    def is_valid_bbox_position(self, viewer_pos):
        """Check if the click position is valid (in original image coordinates)."""
        if not self.pixmap() or not self.original_image_size:
            return False
            
        orig_pos = self.viewer_to_original_coords(viewer_pos)
        
        return (0 <= orig_pos.x() <= self.original_image_size.width() - BBOX_SIZE and
                0 <= orig_pos.y() <= self.original_image_size.height() - BBOX_SIZE)

    def get_bbox_at_position(self, viewer_pos):
        """Find the bbox located at the given viewer coordinates."""
        orig_pos = self.viewer_to_original_coords(viewer_pos)
        
        for i, bbox_pos in enumerate(self.bbox_positions):
            bbox_rect = QRect(bbox_pos, QSize(BBOX_SIZE, BBOX_SIZE))
            if bbox_rect.contains(orig_pos):
                return i
        return None

    def mousePressEvent(self, event: QMouseEvent):
        clicked_bbox = self.get_bbox_at_position(event.pos())
        
        if event.button() == Qt.RightButton and clicked_bbox is not None:
            # Delete the bbox with right-click
            del self.bbox_positions[clicked_bbox]
            self.update()
            print(f"Removed Bounding Box {clicked_bbox}")
            # Notify users that bbox state has changed
            self.bbox_status_changed.emit(len(self.bbox_positions) > 0)
        
        elif event.button() == Qt.LeftButton:
            if clicked_bbox is not None:
                # Start dragging an existing bbox
                self.selected_bbox = clicked_bbox
                self.bbox_dragging = True
                orig_bbox_pos = self.bbox_positions[clicked_bbox]
                viewer_bbox_pos = self.original_to_viewer_coords(orig_bbox_pos)
                self.bbox_drag_offset = event.pos() - viewer_bbox_pos
                self.setCursor(Qt.ClosedHandCursor)
            elif self.bbox_active and len(self.bbox_positions) < 3:
                # Create a new bbox
                if self.is_valid_bbox_position(event.pos()):
                    orig_pos = self.viewer_to_original_coords(event.pos())
                    center_x = orig_pos.x()
                    center_y = orig_pos.y()
                    
                    bbox_position = QPoint(
                        center_x - BBOX_SIZE // 2,
                        center_y - BBOX_SIZE // 2
                    )
                    bbox_position = self.confine_to_image(bbox_position)
                    
                    self.bbox_positions.append(bbox_position)
                    print(f"Bounding Box center in original image: (x={center_x}, y={center_y})")
                    self.bbox_active = False
                    self.update()
                    # Notify users that at least one bbox exists
                    self.bbox_status_changed.emit(True)
            elif event.modifiers() == Qt.ControlModifier:
                # Start dragging (panning) the image
                self.dragging = True
                self.drag_start_position = event.pos()
                self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.bbox_dragging and self.selected_bbox is not None:
            new_viewer_pos = event.pos() - self.bbox_drag_offset
            orig_pos = self.viewer_to_original_coords(new_viewer_pos)
            
            center_x = orig_pos.x() + BBOX_SIZE // 2
            center_y = orig_pos.y() + BBOX_SIZE // 2
            
            bbox_position = QPoint(
                center_x - BBOX_SIZE // 2,
                center_y - BBOX_SIZE // 2
            )
            bbox_position = self.confine_to_image(bbox_position)
            self.bbox_positions[self.selected_bbox] = bbox_position
            self.update()
            
        elif self.dragging:
            # Get the current pixmap and viewport sizes
            pixmap = self.pixmap()
            if not pixmap:
                return
                
            viewport = self.rect()
            
             # Compute a new offset based on drag delta
            new_offset = self.offset + (event.pos() - self.drag_start_position)
            
            # Constrain the image so it does not move outside the viewport
            if pixmap.width() > viewport.width():
                new_offset.setX(min(0, max(viewport.width() - pixmap.width(), new_offset.x())))
            else:
                new_offset.setX(max(viewport.width() - pixmap.width(), min(0, new_offset.x())))
                
            if pixmap.height() > viewport.height():
                new_offset.setY(min(0, max(viewport.height() - pixmap.height(), new_offset.y())))
            else:
                new_offset.setY(max(viewport.height() - pixmap.height(), min(0, new_offset.y())))
            
            # Apply the new offset and continue dragging
            self.offset = new_offset
            self.drag_start_position = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self.bbox_dragging and self.selected_bbox is not None:
                bbox_pos = self.bbox_positions[self.selected_bbox]
                center_x = bbox_pos.x() + BBOX_SIZE // 2
                center_y = bbox_pos.y() + BBOX_SIZE // 2
                print(f"Bounding Box {self.selected_bbox} center in original image: (x={center_x}, y={center_y})")
            
            # Reset drag state and cursor
            self.dragging = False
            self.bbox_dragging = False
            self.selected_bbox = None
            self.bbox_drag_offset = None
            self.setCursor(Qt.ArrowCursor)

    def paintEvent(self, event):
        if self.pixmap():
            painter = QPainter(self)
            viewport = self.rect()
            pixmap_rect = QRect(self.offset, self.pixmap().size())

            # Prevent the image from moving too far outside the viewport
            if pixmap_rect.right() < viewport.width():
                self.offset.setX(viewport.width() - self.pixmap().width())
            if pixmap_rect.bottom() < viewport.height():
                self.offset.setY(viewport.height() - self.pixmap().height())
            if pixmap_rect.left() > 0:
                self.offset.setX(0)
            if pixmap_rect.top() > 0:
                self.offset.setY(0)

            painter.drawPixmap(self.offset, self.pixmap())

            # Compute the currently displayed image size
            current_width = self.pixmap().width()
            current_height = self.pixmap().height()

            # Draw bounding boxes (convert original coordinates to viewer coordinates)
            for i, bbox_position in enumerate(self.bbox_positions):
                viewer_bbox_pos = self.original_to_viewer_coords(bbox_position)
                current_bbox_size = int(BBOX_SIZE * current_width / self.original_image_size.width())
                
                bbox_rect = QRect(
                    viewer_bbox_pos,
                    QSize(current_bbox_size, current_bbox_size)
                )
                
                pen = painter.pen()
                pen.setColor(self.colors[i])
                pen.setWidth(3 if i == self.selected_bbox else 1)
                painter.setPen(pen)
                painter.drawRect(bbox_rect)

class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initial_scale = 1.0
        self.initial_size = None

        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 1200, 850)
        
        self.current_image_path = None
        self.image_files = []
        self.current_image_index = 0

        self.label = ImageLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # Create main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QVBoxLayout(main_widget)
        
        # Top area (image area + file list)
        top_layout = QHBoxLayout()
        
        # QListWidget to display file list
        file_list_container = QWidget()
        file_list_layout = QVBoxLayout(file_list_container)
        file_list_layout.setContentsMargins(0, 0, 0, 0)
        
        self.file_list = QListWidget()
        self.file_list.setFixedWidth(300)
        self.file_list.itemClicked.connect(self.on_file_selected)
        
        file_list_layout.addWidget(QLabel("Image Files:"))
        file_list_layout.addWidget(self.file_list)
        
        # Image display area
        image_container = QWidget()
        image_container.setFixedSize(800, 800)
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = ImageLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.label)

        # Add widgets to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(file_list_container)
        splitter.addWidget(image_container)
        
        top_layout.addWidget(splitter)
        
        # Bottom fixed button area
        button_container = QWidget()
        button_container.setFixedHeight(50)
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(10, 0, 10, 0)
        
        # Buttons
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_image)
        self.prev_button.setEnabled(False)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setEnabled(False)
        
        self.add_bbox_button = QPushButton("Add Bounding Box")
        self.add_bbox_button.clicked.connect(self.on_button_click)
        self.add_bbox_button.setEnabled(False)
        
        self.load_dir_button = QPushButton("Load Directory")
        self.load_dir_button.clicked.connect(self.load_directory)

        # Add 'Save Directory' button
        self.save_dir_button = QPushButton("Save Directory")
        self.save_dir_button.clicked.connect(self.set_save_directory)
        self.save_dir_button.setEnabled(True)
        self.save_directory = None

        self.save_shortcut = QShortcut(QKeySequence('Ctrl+S'), self)
        self.save_shortcut.activated.connect(self.save_bbox_coordinates)
        
        # Add buttons to layout
        button_layout.addStretch()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.add_bbox_button)
        button_layout.addWidget(self.load_dir_button)
        button_layout.addWidget(self.save_dir_button)
        button_layout.addStretch()
        
        # Add all widgets to main layout
        main_layout.addLayout(top_layout)
        main_layout.addWidget(button_container)

    def set_save_directory(self):
        """Set the directory where bbox files will be saved."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Save Directory",
            os.getcwd(),  # start from current working directory
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            self.save_directory = directory
            print(f"Save directory set to: {self.save_directory}")

    def save_bbox_coordinates(self):
        """Save bbox coordinates when Ctrl+S is pressed."""
        if not self.save_directory:
            QMessageBox.warning(
                self,
                "Warning",
                "Please set a save directory first!",
                QMessageBox.Ok
            )
            return
                
        if not self.current_image_path or not self.label.bbox_positions:
            return
                
        # Derive the relative path structure from the original image path
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.current_image_path))))
        relative_path = os.path.relpath(self.current_image_path, base_path)
        
        # Build the save subdirectory path
        save_subdir = os.path.dirname(relative_path)
        full_save_dir = os.path.join(self.save_directory, save_subdir)
        
        # Create directory if it doesn't exist
        os.makedirs(full_save_dir, exist_ok=True)
        
        # Create filename (change extension only)
        image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        save_path = os.path.join(full_save_dir, f"{image_name}_bbox.txt")
            
        # Write bbox info
        with open(save_path, 'w') as f:
            for i, pos in enumerate(self.label.bbox_positions):
                center_x = pos.x() + BBOX_SIZE // 2
                center_y = pos.y() + BBOX_SIZE // 2
                f.write(f"{i}, {center_x}, {center_y}\n")
            
        print(f"Saved bbox coordinates to: {save_path}")

    def update_remove_button(self, has_bbox):
        """Update the state of the Remove button."""
        self.remove_bbox_button.setEnabled(has_bbox)

    def load_image(self, image_path):
        """Load an image and display it."""
        self.current_image_path = image_path
        self.original_pixmap = QPixmap(image_path)
        original_size = self.original_pixmap.size()
        
        self.label.set_original_image_size(original_size)
        print(f"Image loaded: {image_path}")
        print(f"Original image size: {original_size.width()}x{original_size.height()}")
        
        # Scale the image to fit the label
        self.scaled_pixmap = self.original_pixmap.scaled(
            self.label.width(), self.label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Store the initial scaled size (once)
        if not self.initial_size:
            self.initial_size = self.scaled_pixmap.size()
            
        self.label.setPixmap(self.scaled_pixmap)
        
        # Update window title
        filename = os.path.basename(image_path)
        self.setWindowTitle(f"Image Viewer - {filename} ({self.current_image_index + 1}/{len(self.image_files)})")
        
        # Select the current item in the file list
        matching_items = self.file_list.findItems(filename, Qt.MatchExactly)
        if matching_items:
            self.file_list.setCurrentItem(matching_items[0])

        # Update button states
        self.add_bbox_button.setEnabled(True)

    def find_png_files(self, directory):
        """Find all PNG files in the directory and its subdirectories."""
        png_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.png'):
                    png_files.append(os.path.join(root, file))
        return sorted(png_files)

    def load_directory(self):
        """Open a directory dialog and load PNG files."""
        # Start from the current image's directory if available; otherwise, use the current working directory
        start_dir = os.path.dirname(self.current_image_path) if self.current_image_path else os.getcwd()
        
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            start_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            self.image_files = self.find_png_files(directory)
            if self.image_files:
                print(f"Found {len(self.image_files)} PNG files in the directory and its subdirectories")
                
                # Update the file list UI
                self.file_list.clear()
                for image_path in self.image_files:
                    self.file_list.addItem(os.path.basename(image_path))
                
                # If the current image exists in the list, sync the index; otherwise, reset to 0
                if self.current_image_path:
                    try:
                        self.current_image_index = self.image_files.index(self.current_image_path)
                    except ValueError:
                        self.current_image_index = 0
                else:
                    self.current_image_index = 0
                
                # Load the image and update button states
                self.load_image(self.image_files[self.current_image_index])
                self.update_navigation_buttons()
                self.add_bbox_button.setEnabled(True)  # Enable bbox button when an image is loaded
            else:
                print("No PNG files found in the selected directory and its subdirectories")

    def update_navigation_buttons(self):
        """Update the enabled state of the Previous/Next buttons."""
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.image_files) - 1)

    def prev_image(self):
        """Navigate to the previous image."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.image_files[self.current_image_index])
            self.update_navigation_buttons()

    def next_image(self):
        """Navigate to the next image."""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image(self.image_files[self.current_image_index])
            self.update_navigation_buttons()

    def on_file_selected(self, item):
        """Handle item selection from the file list."""
        filename = item.text()
        for i, image_path in enumerate(self.image_files):
            if os.path.basename(image_path) == filename:
                self.current_image_index = i
                self.load_image(image_path)
                self.update_navigation_buttons()
                break

    def on_button_click(self):
        if len(self.label.bbox_positions) < 3:
            self.label.bbox_active = True
            self.label.remove_active = False  # Disable Remove mode while in Add mode
            self.label.update()
        else:
            self.add_bbox_button.setEnabled(False)

    def on_remove_button_click(self):
        """Handle 'Remove Bounding Box' button click."""
        if self.label.bbox_positions:  # Activate only if at least one bbox exists
            self.label.remove_active = True
            self.label.bbox_active = False   # Disable Add mode while in Remove mode
            self.label.update()

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoomIn()
            else:
                self.zoomOut()

    def zoomIn(self):
        # Store current mouse position
        cursor_pos = self.label.mapFromGlobal(QCursor.pos())
        
        # Compute relative ratios at the mouse position
        rel_x = (cursor_pos.x() - self.label.offset.x()) / self.scaled_pixmap.width()
        rel_y = (cursor_pos.y() - self.label.offset.y()) / self.scaled_pixmap.height()
        
        # Zoom in
        self.label.zoom_factor *= SCALE_FACTOR
        width = int(self.scaled_pixmap.width() * SCALE_FACTOR)
        height = int(self.scaled_pixmap.height() * SCALE_FACTOR)
        self.scaled_pixmap = self.original_pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Compute a new offset to keep the cursor anchored
        new_x = cursor_pos.x() - (rel_x * self.scaled_pixmap.width())
        new_y = cursor_pos.y() - (rel_y * self.scaled_pixmap.height())
        self.label.offset = QPoint(int(new_x), int(new_y))
        
        self.label.setPixmap(self.scaled_pixmap)

    def zoomOut(self):
        if not self.current_image_path:
            return
            
        # Store current mouse position
        cursor_pos = self.label.mapFromGlobal(QCursor.pos())
        
        # Compute relative ratios at the mouse position
        rel_x = (cursor_pos.x() - self.label.offset.x()) / self.scaled_pixmap.width()
        rel_y = (cursor_pos.y() - self.label.offset.y()) / self.scaled_pixmap.height()
        
        # Compute new size
        new_width = int(self.scaled_pixmap.width() / SCALE_FACTOR)
        new_height = int(self.scaled_pixmap.height() / SCALE_FACTOR)
        
        # Do not shrink below the initial scaled size
        if new_width >= self.initial_size.width() and new_height >= self.initial_size.height():
            self.label.zoom_factor /= SCALE_FACTOR
            self.scaled_pixmap = self.original_pixmap.scaled(
                new_width, new_height,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # Compute a new offset to keep the cursor anchored
            new_x = cursor_pos.x() - (rel_x * self.scaled_pixmap.width())
            new_y = cursor_pos.y() - (rel_y * self.scaled_pixmap.height())
            self.label.offset = QPoint(int(new_x), int(new_y))
            
            self.label.setPixmap(self.scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageWindow()  # Remove initial image path
    window.show()
    sys.exit(app.exec_())