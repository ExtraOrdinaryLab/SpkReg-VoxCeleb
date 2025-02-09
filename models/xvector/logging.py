import logging


# Function to convert HEX to ANSI 24-bit escape code
def hex_to_ansi(hex_color, is_background=False):
    """Convert a hex color code to an ANSI escape sequence."""
    hex_color = hex_color.lstrip("#")  # Remove '#' if present
    if len(hex_color) != 6:
        raise ValueError("Invalid hex color format. Use #RRGGBB.")

    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"\033[{48 if is_background else 38};2;{r};{g};{b}m"

# Custom log formatter with level-specific colors
class ColoredFormatter(logging.Formatter):
    # Define hex colors per log level
    COLORS = {
        "DEBUG":    {"HEADER": "#1E3A8A", "TIMESTAMP": "#2563EB"},  # Dark Blue / Blue
        "INFO":     {"HEADER": "#166534", "TIMESTAMP": "#22C55E"},  # Dark Green / Green
        "WARNING":  {"HEADER": "#92400E", "TIMESTAMP": "#FACC15"},  # Dark Yellow / Yellow
        "ERROR":    {"HEADER": "#7F1D1D", "TIMESTAMP": "#EF4444"},  # Dark Red / Red
        "CRITICAL": {"HEADER": "#581C87", "TIMESTAMP": "#C084FC"},  # Dark Purple / Purple
    }

    def format(self, record):
        # Extract filename and line number
        filename = record.pathname.split("/")[-1]
        line_no = record.lineno
        level_name = record.levelname

        # Choose colors based on log level
        level_colors = self.COLORS.get(level_name, self.COLORS["INFO"])
        header_color = hex_to_ansi(level_colors["HEADER"])
        timestamp_color = hex_to_ansi(level_colors["TIMESTAMP"])
        reset_color = "\033[0m"  # Reset to default terminal color

        # Format header as "[LEVEL|file.py:line]"
        header = f"{header_color}[{level_name}|{filename}:{line_no}]{reset_color}"

        # Format timestamp
        timestamp = f"{timestamp_color}{self.formatTime(record, self.datefmt)}{reset_color}"

        # Format message
        message = f"\033[37m{record.getMessage()}{reset_color}"  # White message

        return f"{header} {timestamp} >> {message}"


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)