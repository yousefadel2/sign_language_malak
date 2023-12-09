# This file is executed on every boot (including wake-boot from deepsleep)
#import esp
#esp.osdebug(None)
#import webrepl
#webrepl.start()
import network
from umqtt.simple import MQTTClient
import camera
import time
import socket
from lcd_i2c import LCD
from machine import I2C, Pin
# Connect to Wi-Fi
#######################


# PCF8574 on 0x50
I2C_ADDR = 0x27     # DEC 39, HEX 0x27
NUM_ROWS = 2
NUM_COLS = 16

# define custom I2C interface, default is 'I2C(0)'
# check the docs of your device for further details and pin infos
i2c = I2C(0, scl=Pin(14), sda=Pin(13), freq=800000)
lcd = LCD(addr=I2C_ADDR, cols=NUM_COLS, rows=NUM_ROWS, i2c=i2c)

lcd.begin()

#########################
# Initialize the camera
camera.init(0, format=camera.JPEG)
camera.framesize(camera.FRAME_SVGA  )
camera.brightness(2)
# -2,2 (default 0). 2 brightness
camera.saturation(-2)
#camera.speffect(camera.EFFECT_NONE  )
#camera.whitebalance(camera.WB_NONE  )

camera.contrast(-2)
#-2,2 (default 0). 2 highcontrast

# quality
camera.quality(10)
# 10-63 lower number means higher quality
from machine import Pin
import time

# GPIO pin connected to the LED
led_pin = Pin(4, Pin.OUT)  # Replace '2' with the actual GPIO pin number

def turn_on_led():
    led_pin.on()

def turn_off_led():
    led_pin.off()
turn_on_led()
  # Keep the LED on for 2 seconds

photo = camera.capture()
time.sleep(2)
turn_off_led()
camera.deinit()

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect('jooo', 'hhhh1111')

# Wait for connections
while not wlan.isconnected():
    pass




def on_message(topic, msg):
    global keep_running
    lcd.print(str(msg))
    keep_running = False
keep_running=True
client = MQTTClient("espcam", "192.168.78.179")
#client2 = MQTTClient("espcam", "192.168.78.179")
#client2.connect()
client.connect()
client.set_callback(on_message)
client.publish("photo", photo)
client.subscribe('sign')
try:
    while keep_running:
        # Check for new messages
        client.check_msg()
        
        # Your other code can go here

        # Sleep for a short duration to avoid excessive checking
        time.sleep(1)

except KeyboardInterrupt:
    # Disconnect from the MQTT broker when the program is interrupted
        client.disconnect()
    
    


