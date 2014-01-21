import crash_on_ipy
import numpy as np
import collada

class Builder:
    MAX_V_NUM = 1000000

    def __init__(self, max_v_num=MAX_V_NUM):
        self.vertices = np.zeros((max_v_num, 3), dtype=np.float32)
        self.normals = np.zeros((max_v_num, 3), dtype=np.float32)
        self.texcoords = np.zeros((max_v_num, 2), dtype=np.float32)
        self.count = 0
        self.mesh = collada.Collada()
        self.nodes = []
        self.materialNode = None

    def add_vertex(self, v, n, t):
        i = self.count
        self.vertices[i] = v
        self.normals[i] = n
        self.texcoords[i] = t
        self.count += 1

    def set_material(self, material):
        self.mesh.materials.append(material)
        self.mesh.effects.append(material.effect)
        self.mesh.images.append(material.effect.diffuse.sampler.surface.image)
        self.materialNode = collada.scene.MaterialNode(material.name, material, inputs=[])

    def emit_object(self, name):
        n = self.count
        assert n % 3 == 0
        vertSrc = collada.source.FloatSource(
            '{}-verts'.format(name), self.vertices[:n], ('X', 'Y', 'Z'))
        normSrc = collada.source.FloatSource(
            '{}-norms'.format(name), self.normals[:n], ('X', 'Y', 'Z'))
        texcSrc = collada.source.FloatSource(
            '{}-texcs'.format(name), self.texcoords[:n], ('S', 'T'))
        geometry = collada.geometry.Geometry(self.mesh, name + '-geometry', name, [
            vertSrc, normSrc, texcSrc])
        inputList = collada.source.InputList()
        inputList.addInput(0, 'VERTEX', '#' + vertSrc.id)
        inputList.addInput(1, 'NORMAL', '#' + normSrc.id)
        inputList.addInput(2, 'TEXCOORD', '#' + texcSrc.id)
        indices = np.arange(0, n).repeat(3)
        triSet = geometry.createTriangleSet(indices, inputList, self.materialNode.symbol)
        geometry.primitives.append(triSet)
        self.mesh.geometries.append(geometry)

        geomNode = collada.scene.GeometryNode(geometry, [self.materialNode])
        node = collada.scene.Node('node-' + name, children=[geomNode])
        self.nodes.append(node)

    def finish(self, save_to=None):
        scene = collada.scene.Scene('mainScene', self.nodes)
        self.mesh.scenes.append(scene)
        self.mesh.scene = scene
        if save_to is not None:
            self.mesh.write(save_to)
        self.count = 0
        self.materialNode = None
        self.nodes = []
        return self.mesh

class Material(collada.material.Material):
    def __init__(self, id, name, image_path):
        image = collada.material.CImage(name + '-id', image_path)
        surface = collada.material.Surface(name + '-surface', image)
        sampler = collada.material.Sampler2D(name + '-sampler', surface)
        effect = collada.material.Effect(
            id + '-effect', [surface, sampler], 'phong',
            diffuse=collada.material.Map(sampler, 'UVMap'),
            specular=(1, 1, 1))
        super().__init__(id, name, effect)


def test():
    builder = Builder()
    builder.set_material(Material('mat0', 'mat0', 'xx.png'))
    builder.add_vertex((0, 0, 0), (0, 0, 1), (0, 0))
    builder.add_vertex((1, 0, 0), (0, 0, 1), (1, 0))
    builder.add_vertex((0, 1, 0), (0, 0, 1), (0, 1))
    builder.emit_object('tri0')
    builder.finish('xx.dae')

    collada.Collada('xx.dae')

