from .coneqp import coneqp

options = {}
#from https://github.com/cvxopt/cvxopt (modified)
def qp(P, q, G = None, h = None, A = None, b = None, callback = None, solver = None,
    kktsolver = None, initvals = None, **kwargs):

    """
    Solves a quadratic program

        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x <= h
                    A*x = b.


    Input arguments.

        P is a n x n dense or sparse 'd' matrix with the lower triangular
        part of P stored in the lower triangle.  Must be positive
        semidefinite.

        q is an n x 1 dense 'd' matrix.

        G is an m x n dense or sparse 'd' matrix.

        h is an m x 1 dense 'd' matrix.

        A is a p x n dense or sparse 'd' matrix.

        b is a p x 1 dense 'd' matrix or None.

        solver is None or 'mosek'.

        The default values for G, h, A and b are empty matrices with
        zero rows.


    Output arguments (default solver).

        Returns a dictionary with keys 'status', 'x', 's', 'y', 'z',
        'primal objective', 'dual objective', 'gap', 'relative gap',
        'primal infeasibility, 'dual infeasibility', 'primal slack',
        'dual slack'.

        The 'status' field has values 'optimal' or 'unknown'.
        If the status is 'optimal', 'x', 's', 'y', 'z' are an approximate
        solution of the primal and dual optimal solutions

            G*x + s = h,  A*x = b
            P*x + G'*z + A'*y + q = 0
            s >= 0, z >= 0
            s'*z = o.

        If the status is 'unknown', 'x', 's', 'y', 'z' are the last
        iterates before termination.  These satisfy s > 0 and z > 0, but
        are not necessarily feasible.

        The values of the other fields are defined as follows.

        - 'primal objective': the primal objective (1/2)*x'*P*x + q'*x.

        - 'dual objective': the dual objective

              L(x,y,z) = (1/2)*x'*P*x + q'*x + z'*(G*x - h) + y'*(A*x-b).

        - 'gap': the duality gap s'*z.

        - 'relative gap': the relative gap, defined as

              gap / -primal objective

          if the primal objective is negative,

              gap / dual objective

          if the dual objective is positive, and None otherwise.

        - 'primal infeasibility': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s - h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).


        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || P*x + G'*z + A'*y + q || / max(1, ||q||).

        - 'primal slack': the smallest primal slack, min_k s_k.
        - 'dual slack': the smallest dual slack, min_k z_k.

        If the exit status is 'optimal', then the primal and dual
        infeasibilities are guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  The gap is less than
        solvers.options['abstol'] (default 1e-7) or the relative gap is
        less than solvers.options['reltol'] (default 1e-6).

        Termination with status 'unknown' indicates that the algorithm
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, s, z are close to optimal.


    Output arguments (MOSEK solver).

        The return dictionary has two additional fields
        'residual as primal infeasibility certificate' and
        'residual as dual infeasibility certificate', and 'status' field
        can also have the values 'primal infeasible' or 'dual infeasible'.

        If the exit status is 'optimal', the different fields have the
        same meaning as for the default solver, but the the magnitude of
        the residuals and duality gap is controlled by the MOSEK exit
        criteria.  The 'residual as primal infeasibility certificate' and
        'residual as dual infeasibility certificate' are None.

        Status 'primal infeasible'.
        - 'x', 's': None.
        - 'y', 'z' are an approximate certificate of infeasibility

              G'*z + A'*y = 0,  h'*z + b'*y = -1,  z >= 0.

        - 'primal objective': None.
        - 'dual objective': 1.0.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': None.
        - 'dual slack': the smallest dual slack min z_k.
        - 'residual as primal infeasibility certificate': the residual in
          the condition of the infeasibility certificate, defined as

              || G'*z + A'*y || / max(1, ||c||).

        - 'residual as dual infeasibility certificate': None.

        Status 'dual infeasible'.
        - 'x', 's' are an approximate proof of dual infeasibility

              P*x = 0,  q'*x = -1,  G*x + s = 0,  A*x = 0,  s >= 0.

        - 'y', 'z': None.
        - 'primal objective': -1.0.
        - 'dual objective': None.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': the smallest primal slack min_k s_k .
        - 'dual slack': None.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate: the residual in
          the conditions of the infeasibility certificate, defined as
          the maximum of

              || P*x || / max(1, ||q||),
              || G*x + s || / max(1, ||h||),
              || A*x || / max(1, ||b||).


        If status is 'unknown', all the other fields are None.


    Control parameters.

        The control parameters for the different solvers can be modified
        by adding an entry to the dictionary cvxopt.solvers.options.  The
        following parameters control the execution of the default solver.

            options['show_progress'] True/False (default: True)
            options['maxiters'] positive integer (default: 100)
            options['refinement']  positive integer (default: 0)
            options['abstol'] scalar (default: 1e-7)
            options['reltol'] scalar (default: 1e-6)
            options['feastol'] scalar (default: 1e-7).

        The MOSEK parameters can me modified by adding an entry
        options['mosek'], containing a dictionary with MOSEK
        parameter/value pairs, as described in the MOSEK documentation.

        Options that are not recognized are replaced by their default
        values.
    """

    options = kwargs.get('options',globals()['options'])

    from cvxopt import base, blas
    from cvxopt.base import matrix, spmatrix

    if solver == 'mosek':
        from cvxopt import misc
        try:
            from cvxopt import msk
            import mosek
        except ImportError: raise ValueError("invalid option "\
            "(solver='mosek'): cvxopt.msk is not installed")

        opts = options.get('mosek',None)
        if opts:
            solsta, x, z, y = msk.qp(P, q, G, h, A, b, options=opts)
        else:
            solsta, x, z, y = msk.qp(P, q, G, h, A, b)

        n = q.size[0]
        if G is None: G = spmatrix([], [], [], (0,n), 'd')
        if h is None: h = matrix(0.0, (0,1))
        if A is None: A = spmatrix([], [], [], (0,n), 'd')
        if b is None: b = matrix(0.0, (0,1))
        m = G.size[0]

        resx0 = max(1.0, blas.nrm2(q))
        resy0 = max(1.0, blas.nrm2(b))
        resz0 = max(1.0, blas.nrm2(h))

        if solsta in (mosek.solsta.optimal, getattr(mosek.solsta,'near_optimal',None)):
            if solsta is mosek.solsta.optimal: status = 'optimal'
            else: status = 'near optimal'

            s = matrix(h)
            base.gemv(G, x, s, alpha = -1.0, beta = 1.0)

            # rx = q + P*x + G'*z + A'*y
            # pcost = 0.5 * x'*P*x + q'*x
            rx = matrix(q)
            base.symv(P, x, rx, beta = 1.0)
            pcost = 0.5 * (blas.dot(x, rx) + blas.dot(x, q))
            base.gemv(A, y, rx, beta = 1.0, trans = 'T')
            base.gemv(G, z, rx, beta = 1.0, trans = 'T')
            resx = blas.nrm2(rx) / resx0

            # ry = A*x - b
            ry = matrix(b)
            base.gemv(A, x, ry, alpha = 1.0, beta = -1.0)
            resy = blas.nrm2(ry) / resy0

            # rz = G*x + s - h
            rz = matrix(0.0, (m,1))
            base.gemv(G, x, rz)
            blas.axpy(s, rz)
            blas.axpy(h, rz, alpha = -1.0)
            resz = blas.nrm2(rz) / resz0

            gap = blas.dot(s, z)
            dcost = pcost + blas.dot(y, ry) + blas.dot(z, rz) - gap
            if pcost < 0.0:
                relgap = gap / -pcost
            elif dcost > 0.0:
                relgap = gap / dcost
            else:
                relgap = None

            dims = {'l': m, 's': [], 'q': []}
            pslack = -misc.max_step(s, dims)
            dslack = -misc.max_step(z, dims)

            pres, dres = max(resy, resz), resx
            pinfres, dinfres = None, None

        elif solsta == mosek.solsta.prim_infeas_cer:
            status = 'primal infeasible'

            hz, by = blas.dot(h, z),  blas.dot(b, y)
            blas.scal(1.0 / (-hz - by), y)
            blas.scal(1.0 / (-hz - by), z)

            # rx = -A'*y - G'*z
            rx = matrix(0.0, (q.size[0],1))
            base.gemv(A, y, rx, alpha = -1.0, trans = 'T')
            base.gemv(G, z, rx, alpha = -1.0, beta = 1.0, trans = 'T')
            pinfres =  blas.nrm2(rx) / resx0
            dinfres = None

            x, s = None, None
            pres, dres = None, None
            pcost, dcost = None, 1.0
            gap, relgap = None, None

            dims = {'l': m, 's': [], 'q': []}
            dslack = -misc.max_step(z, dims)
            pslack = None

        elif solsta == mosek.solsta.dual_infeas_cer:
            status = 'dual infeasible'
            qx = blas.dot(q,x)
            blas.scal(-1.0/qx, x)
            s = matrix(0.0, (m,1))
            base.gemv(G, x, s, alpha=-1.0)
            z, y = None, None

            # rz = P*x
            rx = matrix(0.0, (q.size[0],1))
            base.symv(P, x, rx, beta = 1.0)
            resx = blas.nrm2(rx) / resx0

            # ry = A*x
            ry = matrix(0.0, (b.size[0],1))
            base.gemv(A, x, ry)
            resy = blas.nrm2(ry) / resy0

            # rz = s + G*x
            rz = matrix(s)
            base.gemv(G, x, rz, beta = 1.0)
            resz = blas.nrm2(rz) / resz0

            pres, dres = None, None
            dinfres, pinfres = max(resx, resy, resz), None
            z, y = None, None
            pcost, dcost = -1.0, None
            gap, relgap = None, None

            dims = {'l': m, 's': [], 'q': []}
            pslack = -misc.max_step(s, dims)
            dslack = None

        else:
            status = 'unknown'
            x, s, y, z = None, None, None, None
            pcost, dcost = None, None
            gap, relgap = None, None
            pres, dres = None, None
            pslack, dslack = None, None
            pinfres, dinfres = None, None

        """
        status: This is the status of the optimization, which can be 'optimal', 'unknown', 'primal infeasible', 'dual infeasible', 'singular', or other status depending on the specific solver used.
        x: This is the optimal solution of the optimization problem.
        s: This is the output of the slack variable, S. In a typical problem, this relates to "unused resources" or the difference of left and right-hand side of constraints. If the constraints are Ax = b, then s = b - Ax. For inequality constraints, s = b - Ax should be >= 0.
        y: The vector Y of the Lagrange multipliers for the inequality constraints.
        z: The vector Z of the Lagrange multipliers for the equality constraints.
        primal objective: The optimal value of the primal objective function.
        dual objective: The optimal value of the dual objective function.
        gap: The difference between the primal and dual objectives. It indicates the 'closeness' of the primal and dual problems. For a well-posed problem, the gap will be zero or close to zero at the optimum.
        relative gap: This is the gap divided by the absolute value of the primal objective, often used as a stopping criterion.
        primal infeasibility: Primal infeasibility measures how much the primal constraints are violated.
        dual infeasibility: Dual infeasibility measures how much the dual constraints are violated.
        primal slack: Measures the difference between the left-hand side and right-hand side of the primal constraints.
        dual slack: Measures the difference between the left-hand side and right-hand side of the dual constraints.
        residual as primal infeasibility certificate: A residual measure indicating the extent of primal infeasibility.
        residual as dual infeasibility certificate: A residual measure indicating the extent of dual infeasibility.
        """

        return {'status': status, 'x': x, 's': s, 'y': y, 'z': z,
            'primal objective': pcost, 'dual objective': dcost,
            'gap': gap, 'relative gap': relgap,
            'primal infeasibility': pres, 'dual infeasibility': dres,
            'primal slack': pslack, 'dual slack': dslack,
            'residual as primal infeasibility certificate': pinfres,
            'residual as dual infeasibility certificate': dinfres}

    return coneqp(P, q, G, h, None, A,  b, initvals, kktsolver=kktsolver, callback=callback, options=options)
