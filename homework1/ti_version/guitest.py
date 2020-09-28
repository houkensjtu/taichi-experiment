import taichi as ti

ti.init(default_fp = ti.f32, arch = ti.cpu)

rgb = (0.4, 0.8, 1.0)
clr = ti.rgb_to_hex(rgb)  # 0x66ccff    
gui = ti.GUI("velocity plot", (512,512), background_color=0x000fff)
gui.circle(pos = (0.5,0.5), color = 0xffffff, radius = 100)
gui.text(content = "Ni ma ge cou bi!", pos = (0.5,0.1), font_size = 20, color=clr )
gui.text(content = "Ni ma ge cou bi!", pos = (0.5,0.5), font_size = 20, color=clr )
gui.text(content = "Ni ma ge cou bi!", pos = (0.5,0.8), font_size = 20, color=clr )
gui.show("sample.png")
