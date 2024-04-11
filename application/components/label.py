from kivy.lang import Builder
from kivy.properties import ListProperty, NumericProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout

kv_string = """
<ColorLabel>:
    size: self.minimum_size
    size_hint: None, None
    padding: 1
    canvas.before:
        Color:
            rgba: root.border_color[:3] + [1] if root.selected else root.border_color[:3] + [0] 
        Line:
            rectangle: (self.x, self.y, self.width, self.height)
            width: 3
        Color:
            rgba: root.border_color[:3] + [0.3] 
        Rectangle:
            pos: self.pos
            size: self.size
    orientation: 'vertical'
    Label:
        id: label
        font_size: '25sp'
        padding: 10, 0
        pos_hint: {"center_x": 0.5, "center_y": 0.5}
        size_hint: None, None
        size: self.texture_size
        color: 0, 0, 0, 1
"""

Builder.load_string(kv_string)


class ColorLabel(BoxLayout):
    border_color = ListProperty([0, 0, 0, 0])
    selected = NumericProperty(0)

    annotation_form = ObjectProperty(None)
    label_data = ObjectProperty(None)

    def __init__(self, label, **kwargs):
        super(ColorLabel, self).__init__(**kwargs)
        self.ids.label.text = label.label
        self.border_color = label.color
        self.label_data = label

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            for label in self.annotation_form.ids.choose_container.children:
                if label != self:
                    label.selected = 0

            self.selected = 1 - self.selected

            if self.selected:
                self.annotation_form.selected_label = self.label_data
            else:
                self.annotation_form.selected_label = None

            return True
        return super(ColorLabel, self).on_touch_down(touch)
