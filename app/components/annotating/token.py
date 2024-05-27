from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, ListProperty
from kivy.lang import Builder
from app.components.annotating.word import Word

token_kv_string = """
<Token>:
    orientation: 'vertical'
    size: self.minimum_size
    size_hint: None, None
    canvas.before:
        Color:
            rgba: root.border_color
        Line:
            rectangle: (self.x, self.y, self.width, self.height)
            width: 1.5
    BoxLayout:
        id: labels
        spacing: 10
        size_hint: None, None
        size: self.minimum_size

    BoxLayout:
        size_hint: 1, None
        pos_hint: {'center_x': 0.5}
        size: self.minimum_size
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: root.border_color
            Rectangle:
                pos: self.pos
                size: self.size if root.show_border else (0, 0)
        Label:
            id: label_label
            font_size: '15sp'
            color: 0, 0, 0, 1
            size_hint: None, None
            size: self.texture_size
            pos_hint: {'center_x': 0.5}
            bold: True
"""

Builder.load_string(token_kv_string)


class Token(BoxLayout):
    annotation = ObjectProperty(None)
    border_color = ListProperty([0, 0, 0, 1])
    show_border = False
    update_form_state = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Token, self).__init__(**kwargs)
        for word in self.annotation.words:
            label = Word(word=word)
            self.ids.labels.add_widget(label)
        self.update_label()

    def update_label(self):
        if self.annotation.label:
            self.ids.label_label.text = self.annotation.label.label
            self.show_border = True
            self.border_color = self.annotation.label.color
        else:
            self.ids.label_label.text = " "
            self.show_border = False
            self.border_color = [0, 0, 0, 0]
