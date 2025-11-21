from IPython.display import display, Markdown

TITLES_COLOR = "#669CDD"

def display_text(text: str, level: int = 2, color: str = TITLES_COLOR, center: bool = True, bold: bool = True) -> None:
    align = 'center' if center else 'left'
    strong_open = '<strong>' if bold else ''
    strong_close = '</strong>' if bold else ''

    html = f"<h{level} style='text-align:{align}; color:{color};'>{strong_open}{text}{strong_close}</h{level}>"
    display(Markdown(html))