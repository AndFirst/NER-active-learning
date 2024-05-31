from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.graphics import Color, Line
from app.data_types import ProjectFormState
from app.components.popups.popups import (
    SaveConfirmationPopup,
    ExitConfirmationPopup,
)

kv_string = """
<StatsScreen>:
    GridLayout:
        cols: 2
        size_hint_y: 0.8
        pos_hint: {'top': 1}
        padding: 10

        BoxLayout:
            orientation: 'vertical'
            padding: 20

            Label:
                text: "Occurrences per label"
                size_hint_y: None
                height: 40
                halign: 'center'
                valign: 'middle'
                text_size: self.size
                color: (0, 0, 0, 1)

            GridLayout:
                id: labels_grid
                cols: 2
                padding: 0
                spacing: 0

        BoxLayout:
            orientation: 'vertical'
            padding: 20

            Label:
                text: "Statistics"
                size_hint_y: None
                height: 40
                halign: 'center'
                valign: 'middle'
                text_size: self.size
                color: (0, 0, 0, 1)

            GridLayout:
                id: stats_grid
                cols: 2
                padding: 0

    GridLayout:
        cols: 1
        padding: 20
        size_hint_y: 0.2
        Button:
            id: back_button
            text: 'Back to Main Menu'
            on_release: app.root.current = 'main_menu'
            # canvas.before:
            #     Color:
            #         rgba: 1, 0, 0, 1
            #     Line:
            #         width: 2.
            #         rectangle: self.x, self.y, self.width, self.height
"""

Builder.load_string(kv_string)

class BorderedLabel(Label):
    def __init__(self, **kwargs):
        super(BorderedLabel, self).__init__(**kwargs)
        with self.canvas.before:
            Color(0, 0, 0, 1)
            self.rect = Line(rectangle=(self.x, self.y, self.width, self.height), width=1)
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.rectangle = (self.x, self.y, self.width, self.height)

class StatsScreen(Screen):
    def __init__(self, **kwargs):
        form_state = kwargs.pop("form_state", None)
        super(StatsScreen, self).__init__(**kwargs)
        self.stats = None
        self.labels = []
        self.is_annotation_done = False
        self.form_state: ProjectFormState = form_state

    def on_enter(self):
        self.ids.labels_grid.clear_widgets()
        self.ids.stats_grid.clear_widgets()
        print(self.form_state.labels)
        labels = [label_data.label for label_data in self.form_state.labels]
        if not labels:
            labels = self.labels

        for label in labels:
            self.ids.labels_grid.add_widget(
                BorderedLabel(text=label, color=(0, 0, 0, 1), size_hint_y=None, height=34)
            )
            self.ids.labels_grid.add_widget(
                BorderedLabel(text="0", color=(0, 0, 0, 1), size_hint_y=None, height=34)
            )

        stats_dict = self.get_stats()
        keys = list(stats_dict.keys())
        values = list(stats_dict.values())
        
        keys[0] = "Labeled sentences"
        keys[1] = "Unlabeled sentences"

        for i, key in enumerate(keys):
            if i < 2:
                self.ids.stats_grid.add_widget(
                    BorderedLabel(text=key, font_size=18, color=(0, 0, 0, 1))
                )
                self.ids.stats_grid.add_widget(
                    BorderedLabel(text=str(int(values[i])), font_size=18, color=(0, 0, 0, 1))
                )
            else:
                self.ids.stats_grid.add_widget(
                    BorderedLabel(text=key, font_size=18, color=(0, 0, 0, 1))
                )
                self.ids.stats_grid.add_widget(
                    BorderedLabel(text=f"{values[i]:.2f}", font_size=18, color=(0, 0, 0, 1))
                )

    def get_stats(self):
        return self.stats

    def confirm_exit(self):
        if not self.is_annotation_done:
            exit_confirmation_popup = SaveConfirmationPopup(save_function=self.save)
        else:
            exit_confirmation_popup = ExitConfirmationPopup()
        exit_confirmation_popup.open()
        return True

    def save(self):
        project = self.manager.get_screen("main_menu").project
        project.save(self.save_path)

    def when_annotating_is_done(self):
        self.ids.back_button.parent.remove_widget(self.ids.back_button)
        self.is_annotation_done = True
