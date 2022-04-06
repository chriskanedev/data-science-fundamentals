import numpy as np



class AstralSimulator:
    def __init__(self):        
        self.dt = 0.01
        self.p = 32
        self.s = 8
        self.b = 8.0 / 3.0
        np.random.seed(2018)
        self.sizes = np.exp(np.random.uniform(-18, 4, 40)) * 1e-16
        


    def simulate(self, n, init, corrector=None):
        results = []
        x, y, z = init
        noise = np.random.normal(0, 1, (n, self.sizes.shape[0])) * self.sizes
        noise[100:][noise[100:]<1e-48] = np.nan        
        
        for i in range(n):
            zn = z + self.dt * (x * y - self.b * z)
            yn = y + self.dt * (x * (self.p - z) - y)
            xn = x + self.dt * (self.s * (y - x))
            x, y, z = xn, yn, zn
            
            if corrector:                                               
                x += np.nansum(noise[i,:])
                x = corrector(x, noise[i,:])            
            results.append([x, y, z])
        return results

    def run(self, n, corrector):
        approx = self.simulate(n, [0.5, 0.25, -0.25], corrector)
        true = self.simulate(n, [0.5, 0.25, -0.25])
        return np.array(true), np.array(approx)


from jhwutils.tkanvas import TKanvas
from jhwutils.transformations import euler_matrix


class AstralViewer:
    def __init__(self, approx, true):
        np.random.seed(2018)
        # generate some random stars
        stars = np.random.uniform(-1, 1, (32, 4))
        stars[:, :3] = (stars[:, :3].T / np.sum(stars[:, :3], axis=1).T).T
        stars[:, 3] = 1

        star_size = np.random.uniform(2, 6, (16,))
        self.stars = stars
        self.star_size = star_size
        self.kanvas = TKanvas(draw_fn=self.draw, w=800, h=800)
        self.tstep = 0
        self.approx = approx
        self.true = true

    def draw(self, canvas):
        canvas.clear()
        roll = self.tstep / 1800.0
        
        ## background
        canvas.rectangle(0, 0, canvas.w, canvas.h, fill="gray")
        max_rad = canvas.w
        for radii in [0.45, 0.4, 0.2, 0.1]:
            color = "#" + ("{0:2X}".format(int(255 * (radii)))) * 3

            canvas.circle(
                canvas.cx, canvas.cy, radii * max_rad, outline=color, fill="black"
            )

        ## rotation matrix
        error = np.sum(np.abs(self.approx[self.tstep] - self.true[self.tstep])) * 0.01
        if np.isfinite(error):
            mat = euler_matrix(*(self.approx[self.tstep] - self.true[self.tstep]) * 0.01)
        else:
            mat = euler_matrix(*np.random.uniform(-np.pi, np.pi, 3))

        traj_mat = euler_matrix(0, 0, roll)
        self.tstep += 1
        if self.tstep>=len(self.true):
            self.kanvas.quit(None)

        hor1 = traj_mat @ np.array([-1,0,0,1]) 
        hor2 = traj_mat @ np.array([1,0,0,1]) 

        canvas.line(hor1[0]*canvas.w*2+canvas.cx, hor1[1]*canvas.w*2+canvas.cy, hor2[0]*canvas.w*2+canvas.cx, hor2[1]*canvas.w*2+canvas.cy, fill='gray', width=1)

        

        scale = max_rad * 0.5
        # reference points
        for star, rad in zip(self.stars, self.star_size):
            star = traj_mat @ star
            x, y = star[0] / star[2], star[1] / star[2]
            if star[2] > 0 and np.sqrt(x ** 2 + y ** 2) < 0.9:
                l = 20
                x, y = x * scale + canvas.cx, y * scale + canvas.cy
                canvas.circle(x, y, rad + 1, fill="lightblue")
                canvas.line(x - l, y, x + l, y, fill="lightblue")
                canvas.line(x, y - l, x, y + l, fill="lightblue")

        # projected points
        for star, rad in zip(self.stars, self.star_size):
            star = traj_mat @ mat @ star
            x, y = star[0] / star[2], star[1] / star[2]
            if star[2] > 0 and np.sqrt(x ** 2 + y ** 2) < 0.9:
                canvas.circle(
                    x * scale + canvas.cx, y * scale + canvas.cy, rad, fill="white"
                )

        # text
        canvas.text(canvas.cx, canvas.h-20, text="Target roll: {roll:.1f}".format(roll=np.degrees(roll)), fill='white', font=('Arial', 15))
        if not np.isfinite(error):
            canvas.text(canvas.cx, 30, text="ATTITUDE LOSS".format(error=np.degrees(error)), fill='red', font=('Arial', 20))
        else:
            canvas.text(canvas.cx, 30, text="Attitude error: {error:.1f} degrees".format(error=np.degrees(error)), fill='black', font=('Arial', 20))

