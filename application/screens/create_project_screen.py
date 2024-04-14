from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.boxlayout import BoxLayout

kv_string = """
<CreateProjectScreen>:
    BoxLayout:
        orientation: 'vertical'
        Label:
            color: 0, 0, 0, 1
            text: 'Project info:'
        BoxLayout:
            orientation: 'vertical'
            padding: (100, 10)
            spacing: 20
            TextInput:
                id: name_input
                hint_text: 'Project name'
                multiline: False
            TextInput:
                id: description_input
                hint_text: 'Description'
            Button:
                id: path_button
                text: 'Save project path'
                on_release: root.open_filechooser()
        PrevNextButtons:
            id: prev_next_buttons
            back_text: "Wstecz"
            next_text: "Dalej"
"""

Builder.load_string(kv_string)

class CreateProjectScreen(Screen):
    def __init__(self, **kwargs):
        shared_data = kwargs.pop('shared_data', None)
        super(CreateProjectScreen, self).__init__(**kwargs)

        self.shared_data = shared_data
        self.ids.prev_next_buttons.on_back = self.go_to_welcome
        self.ids.prev_next_buttons.on_next = self.save_and_go_to_data_set

    def go_to_welcome(self):
        self.manager.current = 'welcome'

    def open_filechooser(self):
        filechooser = FileChooserIconView(dirselect=True)
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(filechooser)
        button = Button(text='OK', size_hint=(1, 0.2))
        layout.add_widget(button)
        popup = Popup(title='Choose folder', content=layout,
                      size_hint=(0.9, 0.9))
        button.bind(on_release=lambda x: self.select_path(filechooser, filechooser.selection, popup))
        popup.open()

    def select_path(self, instance, selection, popup):
        if selection:
            self.ids.path_button.text = selection[0]
            popup.dismiss()
    def save_and_go_to_data_set(self):
        name = self.ids.name_input.text.strip()
        description = self.ids.description_input.text.strip()
        path = self.ids.path_button.text.strip()

        if name and description and path:
            self.shared_data.set_data('project_name', name)
            self.shared_data.set_data('project_description', description)
            self.shared_data.set_data('project_path', path)
            self.manager.current = 'data_set'
        else:
            popup = Popup(title='Error',
                          content=Label(
                              text='Provide project information'),
                          size_hint=(None, None), size=(300, 200))
            popup.open()