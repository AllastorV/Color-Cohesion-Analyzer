"""
Color Cohesion Analyzer - Blueprint Theme
Professional dark mode styling with blueprint aesthetic
"""

# Blueprint color scheme
COLORS = {
    # Primary backgrounds
    "bg_primary": "#0d1117",
    "bg_secondary": "#161b22",
    "bg_tertiary": "#21262d",
    "bg_elevated": "#2d333b",
    
    # Grid and lines
    "grid_primary": "#1c2128",
    "grid_secondary": "#252c35",
    "grid_accent": "#30363d",
    
    # Blueprint accent colors
    "accent_blue": "#58a6ff",
    "accent_cyan": "#56d4dd",
    "accent_purple": "#bc8cff",
    
    # Status colors
    "success": "#3fb950",
    "warning": "#d29922",
    "error": "#f85149",
    "info": "#58a6ff",
    
    # Text
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "text_muted": "#6e7681",
    
    # Node colors
    "node_bg": "#1c2128",
    "node_border": "#30363d",
    "node_selected": "#58a6ff",
    "node_hover": "#2d333b",
    
    # Connection lines
    "connection_default": "#484f58",
    "connection_strong": "#58a6ff",
    "connection_weak": "#6e7681",
    "connection_outlier": "#f85149",
}

# Main application stylesheet
STYLESHEET = """
/* Main Window */
QMainWindow {
    background-color: #0d1117;
}

/* Central Widget */
QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 13px;
}

/* Menus */
QMenuBar {
    background-color: #161b22;
    border-bottom: 1px solid #21262d;
    padding: 4px;
}

QMenuBar::item {
    padding: 6px 12px;
    background: transparent;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background: #21262d;
}

QMenu {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 32px 8px 16px;
    border-radius: 4px;
}

QMenu::item:selected {
    background: #21262d;
}

QMenu::separator {
    height: 1px;
    background: #30363d;
    margin: 4px 8px;
}

/* Toolbar */
QToolBar {
    background: #161b22;
    border: none;
    spacing: 8px;
    padding: 8px;
}

QToolButton {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 16px;
    color: #e6edf3;
}

QToolButton:hover {
    background: #2d333b;
    border-color: #484f58;
}

QToolButton:pressed {
    background: #30363d;
}

QToolButton:checked {
    background: #1f6feb;
    border-color: #58a6ff;
}

/* Push Buttons */
QPushButton {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 16px;
    color: #e6edf3;
    font-weight: 500;
}

QPushButton:hover {
    background: #2d333b;
    border-color: #484f58;
}

QPushButton:pressed {
    background: #30363d;
}

QPushButton:disabled {
    background: #161b22;
    color: #6e7681;
    border-color: #21262d;
}

QPushButton#primaryButton {
    background: #238636;
    border-color: #238636;
}

QPushButton#primaryButton:hover {
    background: #2ea043;
}

QPushButton#dangerButton {
    background: #da3633;
    border-color: #da3633;
}

QPushButton#dangerButton:hover {
    background: #f85149;
}

/* Scroll Areas */
QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: #161b22;
    width: 12px;
    margin: 0;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: #30363d;
    min-height: 30px;
    border-radius: 6px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background: #484f58;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #161b22;
    height: 12px;
    margin: 0;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background: #30363d;
    min-width: 30px;
    border-radius: 6px;
    margin: 2px;
}

QScrollBar::handle:horizontal:hover {
    background: #484f58;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* Splitters */
QSplitter::handle {
    background: #30363d;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* Dock Widgets */
QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget::title {
    background: #161b22;
    padding: 8px;
    border-bottom: 1px solid #30363d;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #30363d;
    border-radius: 6px;
    background: #161b22;
}

QTabBar::tab {
    background: #161b22;
    border: 1px solid #30363d;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background: #21262d;
    border-bottom-color: #21262d;
}

QTabBar::tab:hover:!selected {
    background: #1c2128;
}

/* List Widget */
QListWidget {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px;
}

QListWidget::item {
    padding: 8px;
    border-radius: 4px;
}

QListWidget::item:selected {
    background: #1f6feb;
}

QListWidget::item:hover:!selected {
    background: #21262d;
}

/* Tree Widget */
QTreeWidget {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px;
}

QTreeWidget::item {
    padding: 6px;
    border-radius: 4px;
}

QTreeWidget::item:selected {
    background: #1f6feb;
}

QTreeWidget::item:hover:!selected {
    background: #21262d;
}

QTreeWidget::branch {
    background: transparent;
}

/* Line Edit */
QLineEdit {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e6edf3;
    selection-background-color: #1f6feb;
}

QLineEdit:focus {
    border-color: #58a6ff;
}

QLineEdit:disabled {
    background: #161b22;
    color: #6e7681;
}

/* Combo Box */
QComboBox {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e6edf3;
}

QComboBox:hover {
    background: #2d333b;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    selection-background-color: #1f6feb;
}

/* Spin Box */
QSpinBox, QDoubleSpinBox {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e6edf3;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #58a6ff;
}

/* Slider */
QSlider::groove:horizontal {
    height: 4px;
    background: #30363d;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    background: #58a6ff;
    border-radius: 8px;
    margin: -6px 0;
}

QSlider::handle:horizontal:hover {
    background: #79c0ff;
}

/* Progress Bar */
QProgressBar {
    background: #21262d;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background: #58a6ff;
    border-radius: 4px;
}

/* Check Box */
QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #30363d;
    border-radius: 4px;
    background: #0d1117;
}

QCheckBox::indicator:checked {
    background: #1f6feb;
    border-color: #1f6feb;
}

QCheckBox::indicator:hover {
    border-color: #58a6ff;
}

/* Radio Button */
QRadioButton {
    spacing: 8px;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #30363d;
    border-radius: 9px;
    background: #0d1117;
}

QRadioButton::indicator:checked {
    background: #1f6feb;
    border-color: #1f6feb;
}

/* Group Box */
QGroupBox {
    font-weight: 600;
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
}

/* Status Bar */
QStatusBar {
    background: #161b22;
    border-top: 1px solid #21262d;
    padding: 4px;
}

QStatusBar::item {
    border: none;
}

/* Tool Tips */
QToolTip {
    background: #2d333b;
    border: 1px solid #484f58;
    border-radius: 6px;
    padding: 8px;
    color: #e6edf3;
}

/* Label Styles */
QLabel#header {
    font-size: 18px;
    font-weight: 600;
    color: #e6edf3;
}

QLabel#subheader {
    font-size: 14px;
    color: #8b949e;
}

QLabel#metric {
    font-size: 24px;
    font-weight: 700;
    color: #58a6ff;
}

QLabel#metricLabel {
    font-size: 11px;
    color: #6e7681;
    text-transform: uppercase;
}

/* Frame Styles */
QFrame#card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
}

QFrame#separator {
    background: #30363d;
    max-height: 1px;
}
"""

# Node-specific styles
NODE_STYLE = {
    "default": {
        "background": "#1c2128",
        "border": "#30363d",
        "text": "#e6edf3",
        "border_width": 2,
        "border_radius": 8,
    },
    "consensus": {
        "background": "#0d1117",
        "border": "#58a6ff",
        "text": "#e6edf3",
        "border_width": 3,
        "border_radius": 12,
    },
    "average": {
        "background": "#0d1117",
        "border": "#bc8cff",
        "text": "#e6edf3",
        "border_width": 3,
        "border_radius": 12,
    },
    "outlier": {
        "background": "#1c2128",
        "border": "#f85149",
        "text": "#e6edf3",
        "border_width": 2,
        "border_radius": 8,
    },
    "selected": {
        "background": "#21262d",
        "border": "#58a6ff",
        "text": "#e6edf3",
        "border_width": 3,
        "border_radius": 8,
    },
    "video": {
        "background": "#1c2128",
        "border": "#56d4dd",
        "text": "#e6edf3",
        "border_width": 2,
        "border_radius": 8,
    },
    "shot": {
        "background": "#21262d",
        "border": "#484f58",
        "text": "#8b949e",
        "border_width": 1,
        "border_radius": 6,
    },
}

# Connection line styles
CONNECTION_STYLE = {
    "default": {
        "color": "#484f58",
        "width": 2,
        "style": "solid",
    },
    "strong": {
        "color": "#58a6ff",
        "width": 3,
        "style": "solid",
    },
    "weak": {
        "color": "#6e7681",
        "width": 1,
        "style": "dashed",
    },
    "outlier": {
        "color": "#f85149",
        "width": 2,
        "style": "solid",
    },
}

def get_cohesion_color(score: float) -> str:
    """Get color based on cohesion score (0-1)"""
    if score >= 0.8:
        return COLORS["success"]
    elif score >= 0.6:
        return COLORS["accent_blue"]
    elif score >= 0.4:
        return COLORS["warning"]
    else:
        return COLORS["error"]

def get_distance_line_style(distance: float, threshold: float = 10.0) -> dict:
    """Get line style based on perceptual distance"""
    normalized = distance / threshold
    
    if normalized < 0.5:
        return CONNECTION_STYLE["strong"]
    elif normalized < 1.0:
        return CONNECTION_STYLE["default"]
    elif normalized < 2.0:
        return CONNECTION_STYLE["weak"]
    else:
        return CONNECTION_STYLE["outlier"]
