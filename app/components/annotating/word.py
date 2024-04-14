from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.lang import Builder


kv_string = """
<Word>:
    text: root.word.word if root.word else ""
    size_hint: None, None
    size: self.texture_size
    color: 0, 0, 0, 1
    font_size: '25sp'
    padding: 10, 0
    pos_hint: {"center_x": 0.5, "center_y": 0.5}
"""

Builder.load_string(kv_string)


class Word(Label):
    word = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Word, self).__init__(**kwargs)

    def on_touch_down(self, touch):
        if self.parent is None or self.parent.parent is None or self.parent.parent.parent is None or self.parent.parent.parent.parent is None:
            return
        form = self.parent.parent.parent.parent
        if self.collide_point(*touch.pos):
            if touch.button == 'left' and form.selected_label:
                if form.multiword_mode:
                    form.update_multi_word_mode(self.word)
                else:
                    form.update_single_word_mode(self.word)
            elif touch.button == 'right':
                if form.multiword_mode:
                    form.commit_multi_label()
                else:
                    form.remove_label(self.word)
            else:
                super().on_touch_down(touch)
