import numpy as np
from PIL import Image
class Material:
    def __init__(self,color=[255,255,255],diffusivity=1,specularity=0,shininess=1,reflectivity = 0.0): #a property of shapes that tell us how to reflect light from the shape.
        self.color = color #color of material
        self.diffusivity = diffusivity #the strength of diffuse reflection of light from this material
        self.specularity = specularity #the strength of specular reflection of light from this material
        self.shininess = shininess #how "concentrated" the specular reflections are around light sources
        self.reflectivity = reflectivity
        
class Ray: #a ray is a line (parametrised in t) which ends on one side. This means that there is a minimum value of t (take it as t = 0 here).
    def __init__(self,source,direction): #source and direction indicate where the ray starts and the direction it travels as t increases.
        self.source = source
        self.direction = direction
        self.x0 = source[0]
        self.x1 = direction[0]
        self.y0 = source[1]
        self.y1 = direction[1]
        self.z0 = source[2]
        self.z1 = direction[2]
    def get_point(self,t):
        return np.array([self.x0+self.x1*t,self.y0+self.y1*t,self.z0+self.z1*t])

class Sphere:
    def __init__(self,centre,radius,material):
        self.centre = centre
        self.radius = radius
        self.material = material
    def get_normal(self,point): #return the vector perpendicular to the sphere (of length 1), at a point presumed to be on the sphere. Points out from sphere.
        x = point[0]
        y = point[1]
        z = point[2]
        return np.array([(x-self.centre[0])/self.radius,(y-self.centre[1])/self.radius,(z-self.centre[2])/self.radius])
        
    def intersects(self,ray):#returns the t-values of the intersections between a ray and a sphere if they exist. If one does not exist, the third and fourth arguments are False.
        a = ray.x1**2 + ray.y1**2 + ray.z1**2
        b = 2*((ray.x1*(ray.x0-self.centre[0])) +(ray.y1*(ray.y0-self.centre[1]))+(ray.z1*(ray.z0-self.centre[2])))
        c = (ray.x0-self.centre[0])**2 +(ray.y0-self.centre[1])**2+(ray.z0-self.centre[2])**2 - (self.radius**2)
        #print(str(a)+", "+str(b)+", "+str(c))
        t_0 = 0
        t_1 = 0
        discriminant = b**2 - (4*a*c)
        if(discriminant < 0): #the ray will not intersect if the quadratic has no real solutions
            return [0,0,False,False] #return t = 0 to prevent crashing, and false to indicate there was no intersection
        else:
            t_0 = (-b - np.sqrt(discriminant))/(2*a) #note that we always have t_0 <= t_1
            t_1 = (-b + np.sqrt(discriminant))/(2*a)
        if(t_0<0 and t_1>0):
            return [0,t_1,False,True]
        if(t_0<0 and t_1>0):
            return [0,0,False,False]
        return [t_0,t_1,True,True]
        
    def get_color_at_point(self,point):
        return self.material.color
        
class Plane:
    def __init__(self,point,normal,material,texture_path = None,second_point = np.array([0,0,0]),tex_size = 100):
        self.point = point
        self.normal = normalise(normal)
        self.d = np.dot(self.point,self.normal)
        self.material = material
        self.texture_image = np.array([0])
        self.second_point = second_point #get the second point so we can define some orthogonal vectors in the plane for texture mapping. This point must also be in the plane (make this user-proof later)
        #second_point seems to work best if it is "above" the first point in the plane (so that second_point-first_point is a vector pointing whichever way "up" is in the texture image)
        
        if(np.array_equal(self.second_point,np.array([0,0,0]))):
            if(normal[0]!=0):
                second_point = (self.d)/normal[0]
            elif(normal[1]!=0):
                second_point = (self.d)/normal[1]
            elif(normal[2]!=0):
                second_point = (self.d)/normal[2]
                
        self.tex_vec_1 = normalise(self.second_point-self.point) #these will be our "coordinate axes" for mapping pixels in a texture file to points on the plane
        self.tex_vec_2 = -np.cross(normal,self.tex_vec_1) #if tex_vec_1 is the x-axis, using the minus sign makes this a y-axis that points up.
        self.tex_size = tex_size #a tex_size of 100 means that 100 pixels in a texture image correspond to 1 unit in the scene space.
        if(not(texture_path is None)):
            try:
                self.texture_image = np.asarray(Image.open(texture_path,mode="r"))
            except FileNotFoundError:
                print("File not found: "+texture_path+", continuing using default colour of object")
    def get_normal(self,point):
        return self.normal
        
    def get_color_at_point(self,color_point):
        if(not(np.array_equal(self.texture_image,np.array([0])))): #if the texture_image array is different from its default value, use that texture rather than the color of the underlying material
        
            difference_vector = color_point-self.point #gives the position of the color_point "relative" to the point that defines the plane
                     
            x_coord = int(np.dot(difference_vector,self.tex_vec_1)*self.tex_size) #taking the dot product of the difference vector with the two "texture vectors" gives the point's coordinates in the orthonormal basis
            y_coord = int(np.dot(difference_vector,self.tex_vec_2)*self.tex_size)
            img_length = np.shape(self.texture_image)[0]
            img_height = np.shape(self.texture_image)[1]
            return self.texture_image[x_coord%img_length][y_coord%img_height] #take the coordinates modulo length and height to make the texture repeat
            
        return self.material.color
        
    def intersects(self,ray):
        if (np.dot(ray.direction,self.normal)==0):
            return [0,0,False,False]
        t_int = (self.d - np.dot(self.normal,ray.source))/np.dot(self.normal,ray.direction)
        if(t_int<0):
            return [0,0,False,False]
        return [t_int,t_int,True,True]
        
    def contains(self,check_point):
        if np.dot(check_point,self.normal)<=self.d:
            return True
        return False

class SphereSlice: #a sphere with some of the surface cut away. Useful to make mirrors, specifically concave ones.        
    def __init__(self,edge_centre,radius,pole_dir,max_edge_dist,material):
        self.edge_centre = edge_centre
        self.radius = radius
        self.material = material
        self.unit_pole = normalise(pole_dir)# a unit vector that points in the direction of the edge of the sphere (at the centre of the slice) to the centre of the actual sphere.
        self.pole = self.unit_pole*self.radius #by multiplying the unit pole by the radius, we get a vector that is actually *equal* to the centre of the sphere minus the centre of the slice.
        self.sphere_centre = self.edge_centre+self.pole #the centre of the underlying sphere.
        self.max_edge_dist = max_edge_dist #the distance from the point at the center of the slice, towards the sphere center, at which the slice gets cut off. 
        #E.g. setting max_edge_dist to 1 for a sphere with radius 1 would give a half-sphere.
        self.cutoff_plane = Plane(self.edge_centre+(self.unit_pole*self.max_edge_dist),self.pole,self.material) #a plane separating the slice from the rest of the sphere
        
    def intersects(self,ray):
        a = ray.x1**2 + ray.y1**2 + ray.z1**2
        b = 2*((ray.x1*(ray.x0-self.sphere_centre[0])) +(ray.y1*(ray.y0-self.sphere_centre[1]))+(ray.z1*(ray.z0-self.sphere_centre[2])))
        c = (ray.x0-self.sphere_centre[0])**2 +(ray.y0-self.sphere_centre[1])**2+(ray.z0-self.sphere_centre[2])**2 - (self.radius**2)
        
        discriminant = b**2 - (4*a*c)
        t_0 = 0
        t_1 = 0
        if(discriminant < 0): #the ray will not intersect if the quadratic has no real solutions
            return [0,0,False,False] #return t = 0 to prevent crashing, and false to indicate there was no intersection

        t_0 = (-b - np.sqrt(discriminant))/(2*a) #note that we always have t_0 <= t_1
        t_1 = (-b + np.sqrt(discriminant))/(2*a)

        cutoff_0 = self.cutoff_plane.contains(ray.get_point(t_0)) #use the cutoff plane to check if these t-values should be rendered as part of the slice
        cutoff_1 = self.cutoff_plane.contains(ray.get_point(t_1))
        
        if(cutoff_0 and cutoff_1):
            return [t_0,t_1,True,True]
        elif(cutoff_0 and (not cutoff_1)):
            return [t_0,0,True,False]
        elif((not cutoff_0) and cutoff_1):
            return [0,t_1,False,True]
        else:
            return [0,0,False,False]
    def get_normal(self,point):
        x = point[0]
        y = point[1]
        z = point[2]
        return np.array([(x-self.sphere_centre[0])/self.radius,(y-self.sphere_centre[1])/self.radius,(z-self.sphere_centre[2])/self.radius])
    def get_color_at_point(self,point):
        return self.material.color
        
def normalise(v):
    return v/np.linalg.norm(v)