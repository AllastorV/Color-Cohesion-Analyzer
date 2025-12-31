"""
Color Cohesion Analyzer - Main Window
Professional blueprint-style interface
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QToolBar, QPushButton, QLabel, QFileDialog,
    QStatusBar, QProgressBar, QMessageBox, QMenu, QMenuBar,
    QApplication, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QAction, QIcon, QFont, QDragEnterEvent, QDropEvent, QShortcut, QKeySequence

from .theme import STYLESHEET, COLORS
from .node_graph import NodeGraphView
from .panels import SidebarPanel, AssetMetricsPanel
from .translations import t, toggle_language, get_current_language, add_language_change_listener
from config import (
    ANALYSIS_CONFIG, EXPORT_CONFIG, 
    is_image_file, is_video_file, is_supported_file
)


class AnalysisWorker(QThread):
    """Background worker for analysis tasks"""
    
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    paused_changed = pyqtSignal(bool)  # Signal for pause state changes
    
    def __init__(self, files: List[str], use_gpu: bool = False):
        super().__init__()
        self.files = files
        self.use_gpu = use_gpu
        self._paused = False
        self._stopped = False
        self._mutex = None
        self._pause_condition = None
    
    def pause(self):
        self._paused = True
        self.paused_changed.emit(True)
    
    def resume(self):
        self._paused = False
        self.paused_changed.emit(False)
    
    def stop(self):
        self._stopped = True
        self._paused = False  # Unblock if paused
    
    def run(self):
        try:
            from core import (
                PaletteExtractor, PaletteAggregator, VideoProcessor,
                MetricsEngine, Palette
            )
            import cv2
            import numpy as np
            
            extractor = PaletteExtractor()
            aggregator = PaletteAggregator()
            video_processor = VideoProcessor()
            metrics_engine = MetricsEngine()
            
            results = {
                "palettes": [],
                "video_analyses": [],
                "asset_metrics": [],
            }
            
            total_files = len(self.files)
            
            for i, file_path in enumerate(self.files):
                # Non-blocking pause check - check periodically instead of tight loop
                while self._paused and not self._stopped:
                    QThread.msleep(100)  # Short sleep to prevent CPU spinning
                    if self._stopped:
                        break
                
                if self._stopped:
                    break
                
                filename = Path(file_path).name
                self.progress.emit((i / total_files) * 100, f"Processing: {filename}")
                
                if is_image_file(file_path):
                    # Process image
                    image = cv2.imread(file_path)
                    if image is not None:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        palette = extractor.extract_from_image(image_rgb, filename)
                        results["palettes"].append(palette)
                        aggregator.add_palette(palette)
                
                elif is_video_file(file_path):
                    # Process video - extract single global palette from entire video
                    def progress_callback(prog, msg):
                        overall = (i + prog) / total_files * 100
                        self.progress.emit(overall, f"{filename}: {msg}")
                    
                    video_processor.set_progress_callback(progress_callback)
                    analysis = video_processor.analyze_video(file_path)
                    
                    if analysis:
                        results["video_analyses"].append(analysis)
                        
                        # Only add the global video palette (single palette per video)
                        if analysis.global_palette:
                            results["palettes"].append(analysis.global_palette)
                            aggregator.add_palette(analysis.global_palette)
                        # Note: Shot palettes are NOT added to aggregator
                        # We only want one palette per video for comparison
            
            if self._stopped:
                return
            
            self.progress.emit(90, "Computing consensus and metrics...")
            
            # Compute aggregated palettes
            results["consensus_palette"] = aggregator.compute_consensus()
            results["average_palette"] = aggregator.compute_global_average()
            results["outlier_palette"], results["asset_divergences"] = aggregator.detect_outliers()
            
            # Compute metrics for all palettes
            for palette in results["palettes"]:
                metrics = metrics_engine.compute_asset_metrics(
                    palette, results["consensus_palette"]
                )
                results["asset_metrics"].append(metrics)
            
            # Generate report
            results["report"] = metrics_engine.generate_cohesion_report(
                results["palettes"],
                results["consensus_palette"]
            )
            
            self.progress.emit(100, "Analysis complete")
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class DropZone(QFrame):
    """Initial drop zone for files"""
    
    files_dropped = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 300)
        
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_secondary']};
                border: 2px dashed {COLORS['grid_accent']};
                border-radius: 12px;
            }}
            QFrame:hover {{
                border-color: {COLORS['accent_blue']};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Icon placeholder
        self.icon_label = QLabel(t("drop_zone_title"))
        self.icon_label.setFont(QFont("Segoe UI", 48))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)
        
        # Text
        self.text_label = QLabel(t("drop_zone_text"))
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
        layout.addWidget(self.text_label)
        
        # Supported formats
        self.formats_label = QLabel(t("drop_zone_formats"))
        self.formats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formats_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        layout.addWidget(self.formats_label)
        
        # Register for language changes
        add_language_change_listener(self._update_texts)
    
    def _update_texts(self):
        """Update texts when language changes"""
        self.icon_label.setText(t("drop_zone_title"))
        self.text_label.setText(t("drop_zone_text"))
        self.formats_label.setText(t("drop_zone_formats"))
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(f"""
                QFrame {{
                    background: {COLORS['bg_tertiary']};
                    border: 2px dashed {COLORS['accent_blue']};
                    border-radius: 12px;
                }}
            """)
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_secondary']};
                border: 2px dashed {COLORS['grid_accent']};
                border-radius: 12px;
            }}
        """)
    
    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path) and is_supported_file(path):
                files.append(path)
            elif os.path.isdir(path):
                for root, dirs, filenames in os.walk(path):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        if is_supported_file(file_path):
                            files.append(file_path)
        
        self.dragLeaveEvent(None)
        
        if files:
            self.files_dropped.emit(files)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._browse_files()
    
    def _browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images and Videos",
            "",
            "Media Files (*.jpg *.jpeg *.png *.tiff *.tif *.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)"
        )
        
        if files:
            self.files_dropped.emit(files)


class MainWindow(QMainWindow):
    """
    Main application window for Color Cohesion Analyzer
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle(t("app_title"))
        self.setMinimumSize(1400, 900)
        
        # Set application icon
        self._setup_icon()
        
        # Apply stylesheet
        self.setStyleSheet(STYLESHEET)
        
        # State
        self.current_files: List[str] = []
        self.analysis_results: Optional[Dict] = None
        self.worker: Optional[AnalysisWorker] = None
        
        # Setup UI - order matters: statusbar before toolbar (for gpu_label)
        self._setup_menubar()
        self._setup_statusbar()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_shortcuts()
        
        # Enable drag and drop on main window
        self.setAcceptDrops(True)
        
        # Register for language changes
        add_language_change_listener(self._update_all_texts)
    
    def _setup_icon(self):
        """Setup application icon"""
        # Try multiple icon locations
        icon_paths = [
            Path(__file__).parent.parent / "icon.png",
            Path(__file__).parent.parent / "icon.ico",
            Path(__file__).parent.parent / "assets" / "icon.png",
            Path(__file__).parent.parent / "assets" / "icon.ico",
        ]
        
        for icon_path in icon_paths:
            if icon_path.exists():
                icon = QIcon(str(icon_path))
                self.setWindowIcon(icon)
                # Also set for the application
                QApplication.instance().setWindowIcon(icon)
                break
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # New analysis shortcut
        new_shortcut = QShortcut(QKeySequence("Ctrl+N"), self)
        new_shortcut.activated.connect(self._new_analysis)
    
    def _setup_menubar(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Files...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_files)
        file_menu.addAction(open_action)
        
        open_folder_action = QAction("Open &Folder...", self)
        open_folder_action.triggered.connect(self._open_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        
        export_menu = file_menu.addMenu("&Export")
        
        export_png_action = QAction("Export Palette &PNGs", self)
        export_png_action.triggered.connect(lambda: self._export("png"))
        export_menu.addAction(export_png_action)
        
        export_json_action = QAction("Export &JSON Report", self)
        export_json_action.triggered.connect(lambda: self._export("json"))
        export_menu.addAction(export_json_action)
        
        export_ase_action = QAction("Export &ASE Swatches", self)
        export_ase_action.triggered.connect(lambda: self._export("ase"))
        export_menu.addAction(export_ase_action)
        
        export_all_action = QAction("Export &All...", self)
        export_all_action.triggered.connect(lambda: self._export("all"))
        export_menu.addAction(export_all_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        fit_view_action = QAction("&Fit to Content", self)
        fit_view_action.setShortcut("F")
        fit_view_action.triggered.connect(self._fit_view)
        view_menu.addAction(fit_view_action)
        
        reset_view_action = QAction("&Reset View", self)
        reset_view_action.setShortcut("Home")
        reset_view_action.triggered.connect(self._reset_view)
        view_menu.addAction(reset_view_action)
        
        view_menu.addSeparator()
        
        self.toggle_grid_action = QAction("Show &Grid", self)
        self.toggle_grid_action.setCheckable(True)
        self.toggle_grid_action.setChecked(True)
        self.toggle_grid_action.triggered.connect(self._toggle_grid)
        view_menu.addAction(self.toggle_grid_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        
        self.start_action = QAction("&Start Analysis", self)
        self.start_action.setShortcut("Ctrl+Return")
        self.start_action.triggered.connect(self._start_analysis)
        analysis_menu.addAction(self.start_action)
        
        self.pause_action = QAction("&Pause", self)
        self.pause_action.setEnabled(False)
        self.pause_action.triggered.connect(self._pause_analysis)
        analysis_menu.addAction(self.pause_action)
        
        self.stop_action = QAction("S&top", self)
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self._stop_analysis)
        analysis_menu.addAction(self.stop_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_toolbar(self):
        """Setup toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)
        
        # Add files button
        self.add_btn = QPushButton(t("btn_add_files"))
        self.add_btn.setToolTip(t("tooltip_add_files"))
        self.add_btn.clicked.connect(self._open_files)
        toolbar.addWidget(self.add_btn)
        
        toolbar.addSeparator()
        
        # New analysis button
        self.new_btn = QPushButton(t("btn_new_analysis"))
        self.new_btn.setToolTip(t("tooltip_new_analysis"))
        self.new_btn.clicked.connect(self._new_analysis)
        toolbar.addWidget(self.new_btn)
        
        # Scan controls
        self.start_btn = QPushButton(t("btn_start_scan"))
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.setToolTip(t("tooltip_start_scan"))
        self.start_btn.clicked.connect(self._start_analysis)
        toolbar.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton(t("btn_pause"))
        self.pause_btn.setToolTip(t("tooltip_pause"))
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._pause_analysis)
        toolbar.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton(t("btn_stop"))
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setToolTip(t("tooltip_stop"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_analysis)
        toolbar.addWidget(self.stop_btn)
        
        toolbar.addSeparator()
        
        # Layout controls
        self.layout_label = QLabel(t("label_layout"))
        self.layout_label.setStyleSheet(f"color: {COLORS['text_secondary']}; margin-left: 8px;")
        toolbar.addWidget(self.layout_label)
        
        self.radial_btn = QPushButton(t("btn_radial"))
        self.radial_btn.setToolTip(t("tooltip_radial"))
        self.radial_btn.clicked.connect(self._layout_radial)
        toolbar.addWidget(self.radial_btn)
        
        self.grid_btn = QPushButton(t("btn_grid"))
        self.grid_btn.setToolTip(t("tooltip_grid"))
        self.grid_btn.clicked.connect(self._layout_grid)
        toolbar.addWidget(self.grid_btn)
        
        toolbar.addSeparator()
        
        # GPU/CPU Toggle
        self.gpu_toggle_btn = QPushButton(t("btn_cpu"))
        self.gpu_toggle_btn.setCheckable(True)
        self.gpu_toggle_btn.setToolTip(t("tooltip_gpu_toggle"))
        self.gpu_toggle_btn.clicked.connect(self._toggle_gpu)
        self._update_gpu_button()
        toolbar.addWidget(self.gpu_toggle_btn)
        
        # Spacer
        spacer = QWidget()
        spacer.setFixedWidth(20)
        toolbar.addWidget(spacer)
        
        # Export button
        self.export_btn = QPushButton(t("btn_export"))
        self.export_btn.setToolTip(t("tooltip_export"))
        self.export_btn.clicked.connect(lambda: self._export("all"))
        toolbar.addWidget(self.export_btn)
        
        # Flexible spacer to push language button to right
        flexible_spacer = QWidget()
        flexible_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(flexible_spacer)
        
        # Language toggle button
        self.lang_btn = QPushButton(t("btn_language"))
        self.lang_btn.setToolTip(t("tooltip_language"))
        self.lang_btn.setFixedWidth(60)
        self.lang_btn.clicked.connect(self._toggle_language)
        toolbar.addWidget(self.lang_btn)
    
    def _setup_central_widget(self):
        """Setup central widget with splitter layout"""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Main splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Node graph (center)
        self.graph_container = QWidget()
        graph_layout = QVBoxLayout(self.graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)
        
        # Drop zone (initial state)
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        
        # Node graph view
        self.graph_view = NodeGraphView()
        self.graph_view.hide()
        self.graph_view.node_selected.connect(self._on_node_selected)
        
        graph_layout.addWidget(self.drop_zone)
        graph_layout.addWidget(self.graph_view)
        
        # Sidebar
        self.sidebar = SidebarPanel()
        self.sidebar.setMinimumWidth(320)
        self.sidebar.setMaximumWidth(500)
        self.sidebar.filter_panel.filter_changed.connect(self._on_filter_changed)
        self.sidebar.asset_tree.asset_selected.connect(self._on_asset_selected)
        self.sidebar.filter_panel.reference_combo.currentTextChanged.connect(self._on_reference_changed)
        
        # Add to splitter
        self.splitter.addWidget(self.graph_container)
        self.splitter.addWidget(self.sidebar)
        self.splitter.setSizes([900, 350])
        
        layout.addWidget(self.splitter)
    
    def _setup_statusbar(self):
        """Setup status bar"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Status label
        self.status_label = QLabel(t("ready"))
        self.status_label.setToolTip(t("tooltip_status"))
        self.statusbar.addWidget(self.status_label, 1)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setToolTip(t("tooltip_progress"))
        self.progress_bar.hide()
        self.statusbar.addPermanentWidget(self.progress_bar)
        
        # GPU indicator
        self.gpu_label = QLabel("CPU")
        self.gpu_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.gpu_label.setToolTip(t("tooltip_processing_unit"))
        self.statusbar.addPermanentWidget(self.gpu_label)
        
        # File count
        self.file_count_label = QLabel(t("files_count", count=0))
        self.file_count_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.file_count_label.setToolTip(t("tooltip_file_count"))
        self.statusbar.addPermanentWidget(self.file_count_label)
    
    def _open_files(self):
        """Open file dialog"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            t("dialog_select_files"),
            "",
            "Media Files (*.jpg *.jpeg *.png *.tiff *.tif *.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)"
        )
        
        if files:
            self._on_files_dropped(files)
    
    def _open_folder(self):
        """Open folder dialog"""
        folder = QFileDialog.getExistingDirectory(self, t("dialog_select_folder"))
        
        if folder:
            files = []
            for root, dirs, filenames in os.walk(folder):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    if is_supported_file(file_path):
                        files.append(file_path)
            
            if files:
                self._on_files_dropped(files)
    
    def _on_files_dropped(self, files: List[str]):
        """Handle dropped files"""
        self.current_files.extend(files)
        self.current_files = list(set(self.current_files))  # Remove duplicates
        
        self.file_count_label.setText(t("files_count", count=len(self.current_files)))
        self.status_label.setText(t("status_added_files", count=len(files)))
        
        # Show graph view, hide drop zone
        self.drop_zone.hide()
        self.graph_view.show()
    
    def _start_analysis(self):
        """Start analysis"""
        if not self.current_files:
            QMessageBox.warning(self, t("dialog_no_files"), t("dialog_no_files_msg"))
            return
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.start_action.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_action.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.stop_action.setEnabled(True)
        
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        # Start worker with GPU setting
        use_gpu = ANALYSIS_CONFIG.use_gpu
        self.worker = AnalysisWorker(self.current_files, use_gpu)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.paused_changed.connect(self._on_paused_changed)
        self.worker.start()
    
    def _pause_analysis(self):
        """Pause/resume analysis"""
        if self.worker:
            if self.worker._paused:
                self.worker.resume()
            else:
                self.worker.pause()
    
    def _on_paused_changed(self, is_paused: bool):
        """Handle pause state change from worker"""
        if is_paused:
            self.pause_btn.setText(t("btn_resume"))
            self.pause_btn.setToolTip(t("tooltip_resume"))
            self.status_label.setText(t("status_paused"))
        else:
            self.pause_btn.setText(t("btn_pause"))
            self.pause_btn.setToolTip(t("tooltip_pause"))
            self.status_label.setText(t("status_resumed"))
    
    def _toggle_gpu(self):
        """Toggle GPU/CPU processing"""
        ANALYSIS_CONFIG.use_gpu = not ANALYSIS_CONFIG.use_gpu
        self._update_gpu_button()
    
    def _update_gpu_button(self):
        """Update GPU button state"""
        if ANALYSIS_CONFIG.use_gpu:
            self.gpu_toggle_btn.setText(t("btn_gpu"))
            self.gpu_toggle_btn.setChecked(True)
            self.gpu_toggle_btn.setStyleSheet(f"background-color: {COLORS['success']}; color: white;")
            self.gpu_label.setText("GPU")
            self.gpu_label.setStyleSheet(f"color: {COLORS['success']};")
        else:
            self.gpu_toggle_btn.setText(t("btn_cpu"))
            self.gpu_toggle_btn.setChecked(False)
            self.gpu_toggle_btn.setStyleSheet("")
            self.gpu_label.setText("CPU")
            self.gpu_label.setStyleSheet(f"color: {COLORS['text_muted']};")
    
    def _stop_analysis(self):
        """Stop analysis"""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self._reset_analysis_ui()
            self.status_label.setText(t("status_stopped"))
    
    def _on_progress(self, progress: float, message: str):
        """Handle progress update"""
        self.progress_bar.setValue(int(progress))
        self.status_label.setText(message)
    
    def _on_analysis_finished(self, results: Dict):
        """Handle analysis completion"""
        self.analysis_results = results
        self._reset_analysis_ui()
        self._build_graph()
        self._update_sidebar()
        self.status_label.setText(t("status_complete"))
    
    def _on_analysis_error(self, error: str):
        """Handle analysis error"""
        self._reset_analysis_ui()
        QMessageBox.critical(self, t("dialog_analysis_error"), t("dialog_analysis_error_msg", error=error))
        self.status_label.setText(t("status_error"))
    
    def _reset_analysis_ui(self):
        """Reset UI after analysis"""
        self.start_btn.setEnabled(True)
        self.start_action.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText(t("btn_pause"))
        self.pause_action.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.progress_bar.hide()
    
    def _build_graph(self):
        """Build node graph from analysis results"""
        if not self.analysis_results:
            return
        
        self.graph_view.clear_graph()
        
        results = self.analysis_results
        
        # Add central nodes
        if results.get("consensus_palette"):
            self.graph_view.add_node(results["consensus_palette"], "consensus")
        
        if results.get("average_palette"):
            self.graph_view.add_node(results["average_palette"], "average")
        
        # Add asset nodes
        for palette in results.get("palettes", []):
            # Determine node type
            node_type = "default"
            if palette.source_type == "video":
                node_type = "video"
            elif palette.source_type == "shot":
                node_type = "shot"
            
            # Check if outlier
            for metrics in results.get("asset_metrics", []):
                if metrics.source_name == palette.source_name and metrics.is_outlier:
                    node_type = "outlier"
                    break
            
            self.graph_view.add_node(palette, node_type)
        
        # Add connections
        consensus = results.get("consensus_palette")
        if consensus:
            for palette in results.get("palettes", []):
                from core import compute_palette_distance
                distance = compute_palette_distance(palette, consensus)
                self.graph_view.add_connection(
                    palette.source_name,
                    consensus.source_name,
                    distance
                )
        
        # Layout
        center_names = []
        if results.get("consensus_palette"):
            center_names.append(results["consensus_palette"].source_name)
        if results.get("average_palette"):
            center_names.append(results["average_palette"].source_name)
        
        self.graph_view.layout_radial(center_names)
    
    def _update_sidebar(self):
        """Update sidebar with analysis results"""
        if not self.analysis_results:
            return
        
        results = self.analysis_results
        report = results.get("report", {})
        
        # Update project overview
        proj_data = report.get("project", {})
        if proj_data:
            from core.metrics import ProjectMetrics
            
            # Safely extract nested values with defaults
            composition = proj_data.get("composition", {})
            cohesion = proj_data.get("cohesion", {})
            outliers = proj_data.get("outliers", {})
            temp_dist = proj_data.get("temperature_distribution", {})
            
            metrics = ProjectMetrics(
                total_assets=proj_data.get("total_assets", len(results.get("palettes", []))),
                total_images=composition.get("images", 0),
                total_videos=composition.get("videos", 0),
                total_shots=composition.get("shots", 0),
                average_cohesion=cohesion.get("average", 0),
                cohesion_variance=cohesion.get("variance", 0),
                outlier_count=outliers.get("count", 0),
                outlier_percentage=outliers.get("percentage", 0),
                warm_bias=temp_dist.get("warm_bias", 0),
                cool_bias=temp_dist.get("cool_bias", 0),
                neutral_ratio=temp_dist.get("neutral_ratio", 0),
            )
            
            recommendations = report.get("recommendations", [])
            self.sidebar.project_panel.set_project_metrics(metrics, recommendations)
        else:
            # Calculate basic metrics if report is missing
            palettes = results.get("palettes", [])
            from core.metrics import ProjectMetrics
            
            total_images = sum(1 for p in palettes if p.source_type == "image")
            total_videos = sum(1 for p in palettes if p.source_type == "video")
            
            basic_metrics = ProjectMetrics(
                total_assets=len(palettes),
                total_images=total_images,
                total_videos=total_videos,
            )
            self.sidebar.project_panel.set_project_metrics(basic_metrics, [])
        
        # Update asset tree
        images = []
        videos = []
        
        for palette in results.get("palettes", []):
            if palette.source_type == "image":
                images.append(palette.source_name)
            elif palette.source_type == "video":
                videos.append({"name": palette.source_name, "shots": []})
        
        self.sidebar.asset_tree.set_assets(images, videos)
        
        # Update filter options
        asset_names = [p.source_name for p in results.get("palettes", [])]
        self.sidebar.filter_panel.set_reference_options(asset_names)
    
    def _on_asset_selected(self, asset_name: str):
        """Handle asset selection from tree"""
        if not self.analysis_results:
            return
        
        # Find palette and metrics
        palette = None
        metrics = None
        
        for p in self.analysis_results.get("palettes", []):
            if p.source_name == asset_name:
                palette = p
                break
        
        # Check central palettes
        if not palette:
            if self.analysis_results.get("consensus_palette") and \
               self.analysis_results["consensus_palette"].source_name == asset_name:
                palette = self.analysis_results["consensus_palette"]
            elif self.analysis_results.get("average_palette") and \
                 self.analysis_results["average_palette"].source_name == asset_name:
                palette = self.analysis_results["average_palette"]
        
        for m in self.analysis_results.get("asset_metrics", []):
            if m.source_name == asset_name:
                metrics = m
                break
        
        if palette and metrics:
            self.sidebar.asset_metrics_panel.set_metrics(palette, metrics)
        elif palette:
            # Create basic metrics for central palettes
            from core.metrics import AssetMetrics
            basic_metrics = AssetMetrics(
                source_name=palette.source_name,
                source_type=palette.source_type,
                entropy=palette.entropy
            )
            self.sidebar.asset_metrics_panel.set_metrics(palette, basic_metrics)
    
    def _on_node_selected(self, palette):
        """Handle node selection from graph view"""
        if not palette:
            return
        
        # Update asset metrics panel
        metrics = None
        if self.analysis_results:
            for m in self.analysis_results.get("asset_metrics", []):
                if m.source_name == palette.source_name:
                    metrics = m
                    break
        
        if metrics:
            self.sidebar.asset_metrics_panel.set_metrics(palette, metrics)
        else:
            from core.metrics import AssetMetrics
            basic_metrics = AssetMetrics(
                source_name=palette.source_name,
                source_type=palette.source_type,
                entropy=palette.entropy
            )
            self.sidebar.asset_metrics_panel.set_metrics(palette, basic_metrics)
        
        # Highlight in asset tree
        self.sidebar.asset_tree.select_asset(palette.source_name)
    
    def _on_reference_changed(self, reference_name: str):
        """Handle reference asset change for reference mode"""
        filters = self.sidebar.filter_panel.get_filter_state()
        
        if not filters.get("reference_mode") or not reference_name or not self.analysis_results:
            return
        
        # Find reference palette
        reference_palette = None
        for p in self.analysis_results.get("palettes", []):
            if p.source_name == reference_name:
                reference_palette = p
                break
        
        if not reference_palette:
            return
        
        # Update connections to show distance from reference
        self._update_reference_connections(reference_palette)
    
    def _update_reference_connections(self, reference_palette):
        """Update graph to show distances from reference palette"""
        from core import compute_palette_distance
        
        # Clear existing connections
        for conn in self.graph_view.connections:
            self.graph_view._scene.removeItem(conn)
        self.graph_view.connections.clear()
        
        # Add connections from reference to all other nodes
        for name, node in self.graph_view.nodes.items():
            if name == reference_palette.source_name:
                continue
            if node.node_type in ["consensus", "average"]:
                continue
            
            distance = compute_palette_distance(node.palette, reference_palette)
            self.graph_view.add_connection(
                name,
                reference_palette.source_name,
                distance
            )
        
        # Re-layout with reference as center
        self.graph_view.layout_radial([reference_palette.source_name])
    
    def _on_filter_changed(self):
        """Handle filter changes"""
        # Get filter state
        filters = self.sidebar.filter_panel.get_filter_state()
        
        # Apply display options to graph
        self.graph_view.set_show_connections(filters.get("show_connections", True))
        self.graph_view.set_show_hex(filters.get("show_hex", True))
        self.graph_view.set_compact_mode(filters.get("compact", False))
        
        # Apply view mode filter
        view_mode = filters.get("view_mode", "All Assets")
        self._apply_view_filter(view_mode)
        
        # Handle reference mode
        if filters.get("reference_mode"):
            reference_name = filters.get("reference_asset")
            if reference_name:
                self._on_reference_changed(reference_name)
        else:
            # Reset to normal view with consensus as center
            self._reset_to_consensus_view()
    
    def _apply_view_filter(self, view_mode: str):
        """Filter visible nodes based on view mode"""
        if not self.analysis_results:
            return
        
        for name, node in self.graph_view.nodes.items():
            # Default: show all
            visible = True
            
            if view_mode == "Images Only":
                visible = node.palette.source_type == "image"
            elif view_mode == "Videos Only":
                visible = node.palette.source_type == "video"
            elif view_mode == "Outliers Only":
                # Check if this asset is an outlier
                visible = False
                for m in self.analysis_results.get("asset_metrics", []):
                    if m.source_name == name and m.is_outlier:
                        visible = True
                        break
            
            # Always show central palettes
            if node.node_type in ["consensus", "average"]:
                visible = True
            
            node.setVisible(visible)
        
        # Update connections visibility
        for conn in self.graph_view.connections:
            source_visible = conn.source_node.isVisible()
            target_visible = conn.target_node.isVisible()
            conn.setVisible(source_visible and target_visible)
    
    def _reset_to_consensus_view(self):
        """Reset graph to show connections from consensus palette"""
        if not self.analysis_results:
            return
        
        consensus = self.analysis_results.get("consensus_palette")
        if not consensus:
            return
        
        from core import compute_palette_distance
        
        # Clear existing connections
        for conn in self.graph_view.connections:
            self.graph_view._scene.removeItem(conn)
        self.graph_view.connections.clear()
        
        # Add connections from consensus to all asset nodes
        for palette in self.analysis_results.get("palettes", []):
            distance = compute_palette_distance(palette, consensus)
            self.graph_view.add_connection(
                palette.source_name,
                consensus.source_name,
                distance
            )
        
        # Re-layout with consensus and average as center
        center_names = []
        if self.analysis_results.get("consensus_palette"):
            center_names.append(self.analysis_results["consensus_palette"].source_name)
        if self.analysis_results.get("average_palette"):
            center_names.append(self.analysis_results["average_palette"].source_name)
        
        self.graph_view.layout_radial(center_names)
    
    def _layout_radial(self):
        """Apply radial layout"""
        if self.analysis_results:
            center_names = []
            if self.analysis_results.get("consensus_palette"):
                center_names.append(self.analysis_results["consensus_palette"].source_name)
            if self.analysis_results.get("average_palette"):
                center_names.append(self.analysis_results["average_palette"].source_name)
            self.graph_view.layout_radial(center_names)
    
    def _layout_grid(self):
        """Apply grid layout"""
        self.graph_view.layout_grid(4)
    
    def _fit_view(self):
        """Fit view to content"""
        self.graph_view.fit_to_content()
    
    def _reset_view(self):
        """Reset view"""
        self.graph_view.reset_view()
    
    def _toggle_grid(self):
        """Toggle grid visibility"""
        self.graph_view.show_grid = self.toggle_grid_action.isChecked()
        self.graph_view.viewport().update()
    
    def _export(self, export_type: str):
        """Export analysis results"""
        if not self.analysis_results:
            QMessageBox.warning(self, t("dialog_no_results"), t("dialog_no_results_msg"))
            return
        
        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(self, t("dialog_select_output"))
        
        if not output_dir:
            return
        
        try:
            from export import ProjectExporter
            
            exporter = ProjectExporter(output_dir)
            
            project_path = exporter.export_project(
                "color_analysis",
                self.analysis_results.get("palettes", []),
                self.analysis_results.get("consensus_palette"),
                self.analysis_results.get("average_palette"),
                self.analysis_results.get("outlier_palette"),
                self.analysis_results.get("report", {}),
            )
            
            QMessageBox.information(
                self, t("dialog_export_complete"),
                t("dialog_export_complete_msg", path=project_path)
            )
            
        except Exception as e:
            QMessageBox.critical(self, t("dialog_export_error"), t("dialog_export_error_msg", error=str(e)))
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            t("about_title"),
            t("about_text")
        )
    
    def _new_analysis(self):
        """Clear current analysis and start fresh"""
        # If analysis is running, stop it first
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        # Ask for confirmation if there are results
        if self.analysis_results:
            reply = QMessageBox.question(
                self,
                t("dialog_confirm_new"),
                t("dialog_confirm_new_msg"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Clear state
        self.current_files = []
        self.analysis_results = None
        self.worker = None
        
        # Clear graph
        self.graph_view.clear_graph()
        
        # Reset UI
        self._reset_analysis_ui()
        
        # Show drop zone, hide graph
        self.graph_view.hide()
        self.drop_zone.show()
        
        # Clear sidebar
        self.sidebar.asset_metrics_panel.clear()
        self.sidebar.asset_tree.tree.clear()
        self.sidebar.filter_panel.reference_combo.clear()
        
        # Update status
        self.file_count_label.setText(t("files_count", count=0))
        self.status_label.setText(t("status_cleared"))
    
    def _toggle_language(self):
        """Toggle between English and Turkish"""
        toggle_language()
    
    def _update_all_texts(self):
        """Update all UI texts when language changes"""
        # Window title
        self.setWindowTitle(t("app_title"))
        
        # Toolbar buttons
        self.add_btn.setText(t("btn_add_files"))
        self.add_btn.setToolTip(t("tooltip_add_files"))
        self.new_btn.setText(t("btn_new_analysis"))
        self.new_btn.setToolTip(t("tooltip_new_analysis"))
        self.start_btn.setText(t("btn_start_scan"))
        self.start_btn.setToolTip(t("tooltip_start_scan"))
        
        # Only update pause button if not in paused state
        if not (self.worker and self.worker._paused):
            self.pause_btn.setText(t("btn_pause"))
            self.pause_btn.setToolTip(t("tooltip_pause"))
        else:
            self.pause_btn.setText(t("btn_resume"))
            self.pause_btn.setToolTip(t("tooltip_resume"))
        
        self.stop_btn.setText(t("btn_stop"))
        self.stop_btn.setToolTip(t("tooltip_stop"))
        self.layout_label.setText(t("label_layout"))
        self.radial_btn.setText(t("btn_radial"))
        self.radial_btn.setToolTip(t("tooltip_radial"))
        self.grid_btn.setText(t("btn_grid"))
        self.grid_btn.setToolTip(t("tooltip_grid"))
        self.gpu_toggle_btn.setToolTip(t("tooltip_gpu_toggle"))
        self._update_gpu_button()
        self.export_btn.setText(t("btn_export"))
        self.export_btn.setToolTip(t("tooltip_export"))
        self.lang_btn.setText(t("btn_language"))
        self.lang_btn.setToolTip(t("tooltip_language"))
        
        # Status bar
        self.status_label.setToolTip(t("tooltip_status"))
        self.progress_bar.setToolTip(t("tooltip_progress"))
        self.gpu_label.setToolTip(t("tooltip_processing_unit"))
        self.file_count_label.setText(t("files_count", count=len(self.current_files)))
        self.file_count_label.setToolTip(t("tooltip_file_count"))
        
        # Update sidebar panels (they have their own listeners)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path) and is_supported_file(path):
                files.append(path)
        
        if files:
            self._on_files_dropped(files)


def run_app():
    """Run the application"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
