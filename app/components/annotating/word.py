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
        print(self.parent.parent.parent)
        form = self.parent.parent.parent.parent
        if self.collide_point(*touch.pos):
            if form.selected_label:
                if touch.button == "left":
                    if not form.labels_to_merge:
                        form.labels_to_merge.append(self)
                    elif self.word == form.sentence.get_left_neighbor(
                        form.labels_to_merge[0].word
                    ):
                        form.labels_to_merge.appendleft(self)
                    elif self.word == form.sentence.get_right_neighbor(
                        form.labels_to_merge[-1].word
                    ):
                        form.labels_to_merge.append(self)
                    else:
                        print("Nie da sie dodać tego labela")
                    print(form.labels_to_merge)
                    form.update_labels_to_merge()
            else:
                super().on_touch_down(touch)
