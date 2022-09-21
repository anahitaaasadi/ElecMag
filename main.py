from kivy.app import App
from kivy.properties import OptionProperty, NumericProperty, ListProperty, \
        BooleanProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.clock import Clock
from math import cos, sin

Builder.load_string('''
<ChargePlayground>:
    GridLayout:
        rows: 3
        height: '90dp'    
        size_hint: 1, None
        GridLayout:
            cols: 5
            
            TextInput:
                id: chargevalue
                pos_hint: {'x':root.x , 'y':0}
                width: "60dp"
                hint_text: "insert charge value"
            Button:
                id: chargebutton
                group: 'dashes'
                text: 'submit'
                pos_hint: {'x':root.x , 'y':0}
                width: self.texture_size[0]
                padding_x: '5dp'
                on_press: root.submitcharge(chargevalue.text)
            TextInput:
                id: point_list
                pos_hint: {'x':root.x , 'y':0}
                width: "60dp"
                hint_text: "insert point to remove"
            Button:
                id: run
                group: 'dashes'
                text: 'remove'
                pos_hint: {'x':root.x , 'y':0}
                width: self.texture_size[0]
                padding_x: '5dp'
                on_press: root.remove_canvas(point_list.text)

            Button:
                id: clear
                group: 'dashes'
                text: 'clear'
                pos_hint: {'x':0 , 'y':root.y}
                width: self.texture_size[0]
                padding_x: '5dp'
                on_press: root.clear()
            
                    
        BoxLayout:
            orientation: "horizontal"
            height: 30
    
            BoxLayout:
                orientation: "horizontal"
                size_hint_x: .25
    
                # When clicked the popup opens
                Button:
                    text: "run"
                    on_press: root.open_popup()
            

    

''')
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse
from kivy.uix.popup import Popup
from quadropole import quad
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas,\
                                                NavigationToolbar2Kivy
import matplotlib.pyplot as plt
from kivy.uix.boxlayout import BoxLayout


# Used to display popup
class CustomPopup(Popup):
    fig = plt.figure()
    plot = fig.add_subplot(111)

    def on_plot_hover(self,event):
        # Iterating over each data member plotted
        for curve in self.plot.get_lines():
            # Searching which data member corresponds to current mouse position
            if curve.contains(event)[0]:
                print ("over %s" % curve.get_gid())
                
    def draw(self,ch,pos):
        #fig = plt.figure()
        rect = quad(ch,pos)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_plot_hover)  
        plt.show()


class ChargePlayground(FloatLayout):

    
    points = ListProperty()

    charge = 1
    P = ListProperty()
    Pl = []
    Ll = []
    ch = []
    
    
    def submitcharge(self,chargevalue):
        self.charge = float(chargevalue)
        
    
    
    def clear(self):
        self.P = []
        self.ch = []
        #a = TestLineApp
        for p in self.Pl:
            self.canvas.remove(p)
        for l in self.Ll:
            self.remove_widget(l)
            
            
    def remove_canvas(self,points):
        try:
            points = points.split(",")
            for point in points:
                del self.P[int(point)]
                del self.ch[int(point)]
                self.canvas.remove(self.Pl[int(point)])
                self.remove_widget(self.Ll[int(point)])
        except:
            pass
            
            
    def on_touch_down(self, touch):
        if super(ChargePlayground, self).on_touch_down(touch):
            return True
        ud = touch.ud
        ud['label'] = Label(size_hint=(None, None))
        self.update_touch_label(ud['label'], touch)
        self.add_widget(ud['label'])
        touch.grab(self)
        self.P.append([touch.spos[0]*10, touch.spos[1]*10])
        self.ch.append(self.charge)
        print(touch.spos,self.ch)
        
        self.Ll.append(ud["label"])
        with self.canvas:
            Color(self.charge, 1, 0)
            d = 15.
            a = Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            self.Pl.append(a)
        return True

    def on_touch_move(self, touch):

        if touch.grab_current is self:
            self.P[-1] = [int(touch.x), int(touch.y)]
            return True
        return super(ChargePlayground, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            return True
        return super(ChargePlayground, self).on_touch_up(touch)
    # Opens Popup when called
    def open_popup(self):
        the_popup = CustomPopup()
        #print(self.P[0])
        the_popup.draw(self.ch,self.P)
        #the_popup.open()
    
    def update_touch_label(self, label, touch):
        print(touch.spos)
        label.text = 'ID: %s\nPos: (%d, %d)' % (
            touch.id, touch.spos[0]*10, touch.spos[1]*10)
        label.texture_update()
        label.pos = touch.pos
        label.size = label.texture_size[0] + 10, label.texture_size[1] + 10

    

#class plot(Scatter):
    
class TestLineApp(App):
    def build(self):
        return ChargePlayground()


if __name__ == '__main__':
    TestLineApp().run()
