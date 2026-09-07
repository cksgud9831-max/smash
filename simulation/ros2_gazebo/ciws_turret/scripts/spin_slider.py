#!/usr/bin/env python3
"""Pan + elevation sliders for the CIWS turret -> /turret_controller/commands."""
import math, threading, tkinter as tk
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

PAN=(-180,180); TILT=(0,85)

class Ctl(Node):
    def __init__(self):
        super().__init__('turret_slider')
        self.pub=self.create_publisher(Float64MultiArray,'/turret_controller/commands',10)
        self.pan=0.0; self.tilt=30.0
        self.create_timer(0.05,self._pub)
    def _pub(self):
        m=Float64MultiArray(); m.data=[math.radians(self.pan),math.radians(self.tilt)]; self.pub.publish(m)

def main():
    rclpy.init(); n=Ctl()
    threading.Thread(target=rclpy.spin,args=(n,),daemon=True).start()
    root=tk.Tk(); root.title('CIWS Turret'); root.geometry('360x240')
    sw={'on':False,'d':1}
    def op(v): n.pan=float(v)
    def ot(v): n.tilt=float(v)
    tk.Label(root,text='PAN (deg)',font=('TkDefaultFont',10,'bold')).pack()
    ps=tk.Scale(root,from_=PAN[0],to=PAN[1],orient=tk.HORIZONTAL,length=320,resolution=1,command=op); ps.set(0); ps.pack()
    tk.Label(root,text='ELEVATION (deg)',font=('TkDefaultFont',10,'bold')).pack()
    ts=tk.Scale(root,from_=TILT[0],to=TILT[1],orient=tk.HORIZONTAL,length=320,resolution=1,command=ot); ts.set(30); ts.pack()
    def tog():
        sw['on']=not sw['on']
        b.config(text='Auto-sweep: ON' if sw['on'] else 'Auto-sweep: OFF',relief=tk.SUNKEN if sw['on'] else tk.RAISED)
    b=tk.Button(root,text='Auto-sweep: OFF',command=tog); b.pack(pady=6)
    def tick():
        if sw['on']:
            v=ps.get()+sw['d']*2
            if v>=180: v,sw['d']=180,-1
            elif v<=-180: v,sw['d']=-180,1
            ps.set(v)
        root.after(50,tick)
    tick()
    def close(): n.destroy_node(); rclpy.shutdown(); root.destroy()
    root.protocol('WM_DELETE_WINDOW',close); root.mainloop()

if __name__=='__main__': main()
