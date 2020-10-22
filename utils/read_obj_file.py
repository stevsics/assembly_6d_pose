"""A class that generates mesh data from .obj files."""

class MeshData():
    def __init__(self, obj_file):
        self.vertices = []
        self.faces = []

        with open(obj_file, 'r') as f:
            for line in f:
                parse_line = line.split()
                if parse_line[0] == 'v':
                    self.vertices.append([float(parse_line[1]), float(parse_line[2]), float(parse_line[3])])
                elif parse_line[0] == 'f':
                    new_face = []
                    for it_vert, line_element in enumerate(parse_line):
                        if it_vert == 0: continue
                        face_vert = int(line_element.split('/')[0]) - 1
                        new_face.append(face_vert)
                    self.faces.append(new_face)

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces
