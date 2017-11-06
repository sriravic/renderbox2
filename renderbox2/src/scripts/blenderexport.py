from xml.etree import ElementTree as ET
import bpy

'''
copy and paste from http://effbot.org/zone/element-lib.htm#prettyprint
it basically walks your tree and adds spaces and newlines so the tree is
printed in a nice way
'''
def indent(elem, level=0):
  i = "\n" + level*"  "
  if len(elem):
    if not elem.text or not elem.text.strip():
      elem.text = i + "  "
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
    for elem in elem:
      indent(elem, level+1)
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
  else:
    if level and (not elem.tail or not elem.tail.strip()):
      elem.tail = i

 
'''
Exports the current blender scene as a xml file for renderbox2.
'''
def export_scene():
  
  scene = ET.Element("scene")
  
  camera = ET.SubElement(scene, "camera")
    
  for obj in bpy.data.objects:
      # check if a mesh
      # emissive materials form the lights and we add them to the light list
      if obj.type == 'MESH':
          '''
          get the material params and push it into the material list
          add material reference to the mesh object
          '''
      
      # check if camera
      if obj.type == 'CAMERA':
          position = ET.SubElement(camera, "position")
          position.text = "{" + location.x + "," + location.y + "," + location.z + "}"    
    
  indent(scene)
  
  tree = ET.ElementTree(scene)
  
  tree.write("C:\\temp\\scene.xml", xml_declaration=True, encoding='utf-8', method="xml") 
 
   
'''
main function, so this program can be called by python program.py
'''
if __name__ == "__main__":
  export_scene()