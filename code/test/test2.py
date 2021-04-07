import unittest

class SomeTest(unittest.TestCase):
    def test_isupper(self):
        print("    fff")
        print("    fff".lstrip())
        output = "Kumar_Ravi_003".split("-")
        print("I love Python programming"[7:13])
        print("I love Python programming"[-18:-12])
        print(output)
        print(len("Python"))

        print(" ^ ".join(["Python", "HAHA"]))
        word = [1,2,3,4]
        word[ : ] = [ ]
        print(word)
        print(5.0 / 2)

        t1 = (1,2,3, [34])
        locs = ((1,2), (2,3))
        (l1, l2) = locs
        print('map on a tuple')
        print(list(map(lambda xy: xy[0] + xy[1], locs)))
        print(l1)
        print(l2)
        a = tuple(list(t1)+[5])
        print(a)

        some_dict ={
            'x': 1,
            'y': 2,
            'stuff': {
                'some': 'stuff',
                'f': 1,
                'g': {
                    'a': 3, 'b': 5
                }
            }
        }

        print(some_dict['stuff']['g'])
        print(some_dict['stuff'].get('fgfgfg', "LOL"))
        print(dict(a=1, b=2))
        x=dict(a=1, b=2)
        del x['b']
        print(x)
        print({'a':1, 'b':2})

        set_1 = set([1,2,3,4])
        set_1.symmetric_difference()
