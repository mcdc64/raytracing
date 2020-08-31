# Cian McDonnell 2020
# Feel free to use or edit this code as you wish, but please credit me as the creator.
import numpy as np
from PIL import Image
import time
import threading

from classes import Ray,Sphere,Plane,SphereSlice,Material #all the classes defined for the project in another file

res_y = 1500
res_z = 1500

start = time.time()

screen_y_min = -0.9 #the y and z boundaries of the screen. The screen is normal to the x-axis, so its x coordinate is fixed.
screen_y_max = 0.6

screen_z_min = 8
screen_z_max = 9.5

screen_x = -20

viewpoint = np.array([-24,0,10]) #the viewpoint. The screen should be between the viewpoint and the objects to be rendered.

light_sources = [np.array([-5,5,10])] #the light sources in the scene.
light_strengths = [1]

reflection_limit = 4 #how many times a reflected ray is allowed to "bounce around" before returning the colours it picks up.

white = [255,255,255] #defining colours
coal = [80,80,80]
orange = [240,160,0]
blue = [80,160,225]
red = [220,80,20]
green = [0,200,50]

specular_blue = Material(blue,diffusivity=0.3,specularity=0.8,shininess=200) #defining materials, with colour, diffusivity, specularity, shininess, and reflectivity properties
diffuse_white = Material(white,1,0,1)
specular_red = Material(red,diffusivity=0.7,specularity=0.0,shininess=500)

ground_material = Material(green,diffusivity=0.7,specularity=0.3,shininess=50)
brick = Material(green,diffusivity = 0.7,specularity = 0,shininess = 0)
ref_wall = Material(white,diffusivity = 0.1, specularity = 0,shininess = 0,reflectivity = 0.9)
ball_material = Material(blue,diffusivity=0.1,specularity=0.7,shininess=50,reflectivity = 1)


#defining all the objects that will appear in the scene

sphere1 = Sphere(np.array([1,1,1]),1,ball_material) #define objects that will appear in the scene.
sphere2 = Sphere(np.array([2.5,-1.6,1]),1,ball_material)
ground = Plane(np.array([0,-10,0]),np.array([0,0,1]),ground_material,second_point=np.array([1,-10,0]),texture_path = "textures/wood_1.jpg",tex_size = 50) #can apply textures to planes since the mapping is relatively easy. 
wall1 = Plane(np.array([0,-5,0]),np.array([0,1,0]),brick,second_point = np.array([0,-5,1]),texture_path = "textures/brick.jpg")
wall2 = Plane(np.array([5,0,0]),np.array([-1,0,0]),brick,second_point = np.array([5,0,1]),texture_path = "textures/brick.jpg")


objects = [sphere1,sphere2,ground,wall1,wall2] #all the objects are in a list to make it easy to iterate through them.
    
def normalise(v):
    return v/np.linalg.norm(v)
    



def add_color(prim_ray,intersection_point,intersecting_object,light_source,light_strength,min_prim_t): #calculates what color a pixel should be due to light falling on an object
    global objects
    
    lit = True
    

    shadow_vect = light_source-intersection_point
    shadow_ray = Ray(intersection_point,shadow_vect)
    unit_shadow_vect = normalise(shadow_vect)
    
    
    shadow_traces = [object.intersects(shadow_ray) for object in objects] #can tell us whether the shadow ray hits any objects on its journey to the light source
    
    
    if isinstance(intersecting_object,SphereSlice): #if the object is a SphereSlice, we can sometimes simultaneously see both the inside and outside. Make sure that a given light source does not light up both inside and outside.
        #go "back" along the primary ray to see if we are looking from the inside or outside.
        looking_from_outside = np.linalg.norm(prim_ray.get_point(min_prim_t-(0.01*intersecting_object.radius))-intersecting_object.sphere_centre)>intersecting_object.radius 
        #check which side the light source is on.
        
        lit_from_outside = np.linalg.norm(shadow_ray.get_point(0.01*intersecting_object.radius)-intersecting_object.sphere_centre)>intersecting_object.radius
        if(lit_from_outside ^ looking_from_outside): #xor to check if the viewer is on the opposite side to the light source. If so, the light source can't see that part of the slice.
            lit = False
    
    for shadow_trace in shadow_traces: #this checks the t-values at which the shadow ray hit objects, if it hit any at all. If there is a t-value between 0 and 1, there is an object blocking the light, so set "lit" to False.
        if(shadow_trace[2]):
            if(0.0001<shadow_trace[0] and shadow_trace[0]<1):
                lit = False
        if(shadow_trace[3]):
            if(0.0001<shadow_trace[1] and shadow_trace[1]<1):
                lit = False


    color = np.array([0,0,0],dtype=np.uint8) #set the colour to 0 initially, then change it if the object is lit.
    
    if(lit): #If the primary ray hit a lit part of the object, light it up based on the angle to the light, and its material properties
        normal = normalise(intersecting_object.get_normal(intersection_point))
        reflection_vector = (2*np.dot(unit_shadow_vect,normal)*normal) - unit_shadow_vect #reflection of the shadow vector at the point of incidence (if it were going from light source to surface)

        diffuse_brightness = light_strength*np.abs(np.dot(normal,unit_shadow_vect))*intersecting_object.material.diffusivity #add diffuse brightness
        specular_brightness = light_strength*intersecting_object.material.specularity*max(0,np.dot(reflection_vector,normalise(-prim_ray.direction))**intersecting_object.material.shininess) #specular brightness
        
        red_col = int(min(255,intersecting_object.get_color_at_point(intersection_point)[0]*(diffuse_brightness+specular_brightness)))
        green_col = int(min(255,intersecting_object.get_color_at_point(intersection_point)[1]*(diffuse_brightness+specular_brightness)))
        blue_col = int(min(255,intersecting_object.get_color_at_point(intersection_point)[2]*(diffuse_brightness+specular_brightness)))
        color=[red_col,green_col,blue_col]
    return color
            

def add_reflection(incident_ray,intersection_point,intersecting_object,normal,color_total,count,min_prim_t): #recursive function that bounces light off an object and adds up contributions from reflections that reach that point
    global light_sources,light_strengths,objects
    
    if(count == reflection_limit): #if we reach the maximum allowed number of reflections, stop
        return color_total
        
    min_reflected_t = 10000
    
    min_index = 0
    hit = False
    
    #Note: the normal should have length 1. As of now, all shapes in the program return normals of length 1 so this is OK.
    reflected_vect = incident_ray.direction - 2*np.dot(incident_ray.direction,normal)*normal #vector maths to calculate the vector that is parallel to the reflected ray
    reflected_ray = Ray(intersection_point+(0.001*reflected_vect),reflected_vect) #start off the reflected ray a bit out from the old object (hence the 0.001) to prevent the ray intersecting with the object itself.
    
    hit_infos = [object.intersects(reflected_ray) for object in objects] #checking which object the reflected ray hits first, similar to looking at the primary ray in the "cast_ray" function.
    for info in hit_infos:
        if((info[2] and info[0]>0) or (info[3] and info[0]>0)):
            hit = True
        if(info[0]<=0 and info[1]>0):
            if(info[1]<min_reflected_t):
                min_index = hit_infos.index(info)
                min_reflected_t = info[1]
        elif(info[0]>0):
            if(info[0]<min_reflected_t):
                min_index = hit_infos.index(info)
                min_reflected_t = info[0]
    if(hit):                
        next_point = reflected_ray.get_point(min_reflected_t)#the point hit by the reflected ray. Can recursively see what gets reflected onto THIS point.
        next_object = objects[min_index]
        

    if(not hit):
        return color_total
        
    next_normal = next_object.get_normal(next_point)
    
    for a in range(0,len(light_sources)): #get the colour of the new object at the reflected ray
        new_obj_color = add_color(reflected_ray,next_point,next_object,light_sources[a],light_strengths[a],min_reflected_t)
        for b in range(0,3):
            color_total[b] = color_total[b] + int((intersecting_object.material.reflectivity**(1+count))*new_obj_color[b])
            
    if(not next_object.material.reflectivity==0):#if the next object also happens to reflect, run this function again to see what it shows    
        return add_reflection(reflected_ray,next_point,next_object,next_normal,color_total,count+1,min_reflected_t)
        
    return color_total
    
screen_y_points = np.linspace(screen_y_min,screen_y_max,res_y)
screen_z_points = np.linspace(screen_z_min,screen_z_max,res_z)

pixels = np.full((res_y,res_z,3),127,dtype=np.uint8) #initialise each pixel to grey

def cast_ray(screen_x,screen_y,screen_z,i,j): #cast a ray from a point on the screen, and see if it hits an object in the scene. If so, color the pixel corresponding to that point.
    global pixels
    global viewpoint
    min_prim_t = 10000 #the minimum t-value that the primary ray hits. Initialised to 10000 so that we definitely get a minimum.
    min_index = 0
    screen_point = np.array([screen_x,screen_y,screen_z])
    diff_vector = screen_point - viewpoint #a vector parallel to the primary ray.
    primary_ray = Ray(screen_point,diff_vector)
    
    lit = True
    hit = False
    
    traces = [object.intersects(primary_ray) for object in objects] #where the primary ray intersects each object.
    for trace in traces:
        if((trace[2]and trace[0]>0) or (trace[3] and trace[1]>0)):
            hit = True
        if (trace[0]<=0 and trace[1]>0):
            if (trace[1]<min_prim_t):
                min_index = traces.index(trace)
                min_prim_t = trace[1]
        elif(trace[0]>0):
            if(trace[0]<min_prim_t):
                min_index = traces.index(trace)
                min_prim_t = trace[0]
                
    intersection_point = primary_ray.get_point(min_prim_t) #whatever object is closest to the screen should be the one that gets rendered.
    intersecting_object = objects[min_index]
    if(hit): #adjust color using the "add_color" and "add_reflection" functions.
        pixels[i][j] = 0
        if(not intersecting_object.material.reflectivity == 0):
            reflection_color = add_reflection(primary_ray,intersection_point,intersecting_object,intersecting_object.get_normal(intersection_point),np.array([0,0,0]),0,min_prim_t)
        for k in range(0,len(light_sources)):
            color_to_add = add_color(primary_ray,intersection_point,intersecting_object,light_sources[k],light_strengths[k],min_prim_t)
            
            pixels[i][j][0] = min(255,pixels[i][j][0]+color_to_add[0])
            pixels[i][j][1] = min(255,pixels[i][j][1]+color_to_add[1])
            pixels[i][j][2] = min(255,pixels[i][j][2]+color_to_add[2])
            if(not intersecting_object.material.reflectivity == 0):
                
                for m in range(3):
                    pixels[i][j][m] = min(255,(pixels[i][j][m]+reflection_color[m])) #colour the pixel using the object's own colour and whatever gets added from reflections.
                    if(i%100==0 and j%100==0):
                        print("Added color: "+str(reflection_color[m]))
        if(i%100==0 and j%100==0):
            print(str(i)+", "+str(j)+", "+str(intersection_point)) #print the i and j values every so often so that we know how the render is progressing.

for i in range(0,len(screen_y_points)): #cast rays at all the points on the screen.
    for j in range(0,len(screen_z_points)):
        cast_ray(screen_x,screen_y_points[i],screen_z_points[j],i,j)




trans_pixels = np.transpose(pixels,(1,0,2)) #flip the image around so that the coordinate axes are the right way around in the final image (and so that it is right-side up). They should be right-handed.
output = Image.fromarray(trans_pixels,'RGB')
output = output.rotate(180, Image.NEAREST, expand = 1) #rotate the image.

output.save("output.png")
end = time.time()

print("Took "+str(end-start)+" seconds")
