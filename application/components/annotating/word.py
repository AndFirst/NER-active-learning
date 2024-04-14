from kivy.uix.label import Label
from kivy.lang import Builder

from application.utils import is_key_pressed

kv_string = """
<Word>:
    size_hint: None, None
    size: self.texture_size
    color: 0, 0, 0, 1
    font_size: '25sp'
    padding: 10, 0
    pos_hint: {"center_x": 0.5, "center_y": 0.5}
"""

Builder.load_string(kv_string)


class Word(Label):
    def __init__(self, **kwargs):
        super(Word, self).__init__(**kwargs)

    def on_touch_down(self, touch):
        form = self.parent.parent.parent.parent
        if self.collide_point(*touch.pos):
            if is_key_pressed('ctrl'):
                if touch.button == 'left':
                    form.labels_to_merge.append(self)
                    print('ctrl + left')
                elif touch.button == 'right':
                    form.labels_to_merge.append(self)
                    print('ctrl + right')
            else:
                super().on_touch_down(touch)
