from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup


class ExitConfirmationPopup(Popup):
    def __init__(self, **kwargs):
        super(ExitConfirmationPopup, self).__init__(**kwargs)
        self.title = "Exit Confirmation"
        self.size_hint = (None, None)
        self.size = (300, 200)
        content_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        message_label = Label(text="Are you sure you want to exit?")
        button_layout = BoxLayout(padding=10, spacing=10)
        confirm_button = Button(text="Yes", on_release=self.confirm_exit)
        cancel_button = Button(text="No", on_release=self.dismiss)
        button_layout.add_widget(confirm_button)
        button_layout.add_widget(cancel_button)
        content_layout.add_widget(message_label)
        content_layout.add_widget(button_layout)
        self.content = content_layout

    def confirm_exit(self, instance):
        App.get_running_app().stop()


class SaveConfirmationPopup(Popup):
    def __init__(self, save_function, **kwargs):
        super(SaveConfirmationPopup, self).__init__(**kwargs)
        self.title = "Save Confirmation"
        self.size_hint = (None, None)
        self.size = (300, 200)
        self.save_function = save_function
        self.message_label = Label(text="Do you want to save your data?")
        self.button_layout = BoxLayout(padding=10, spacing=10)
        self.yes_button = Button(text="Yes", on_release=self.save_data)
        self.no_button = Button(text="No", on_release=self.dont_save_data)
        self.cancel_button = Button(text="Cancel", on_release=self.dismiss)
        self.button_layout.add_widget(self.yes_button)
        self.button_layout.add_widget(self.no_button)
        self.button_layout.add_widget(self.cancel_button)
        content_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        content_layout.add_widget(self.message_label)
        content_layout.add_widget(self.button_layout)
        self.content = content_layout

    def save_data(self, instance):
        self.message_label.text = "Saving data..."
        self.button_layout.clear_widgets()
        self.save_function()

        Clock.schedule_once(self.data_saved, 1)

    def data_saved(self, instance):
        self.message_label.text = "Data saved. App will be closed."

        Clock.schedule_once(self.close, 1)

    def close(self, instance):
        App.get_running_app().stop()

    def dont_save_data(self, instance):
        self.message_label.text = "Data not saved. App will be closed."
        self.button_layout.clear_widgets()
        Clock.schedule_once(self.close, 1)
