0.6.0 (Thursday 23rd September 2021)
------------------------------------


* Removed the ``using namespace std`` statement from ``newmat.h``.


0.5.1 (Tuesday 8th June 2021)
-----------------------------


* Added a `feedsRun` test for use with
  [`pyfeeds`](https://git.fmrib.ox.ac.uk/fsl/pyfeeds).


0.5.0 (Tuesday 8th June 2021)
-----------------------------


* Updated to `armadillo` 10.5.1 - this means that C++11 is now the minimum C++
  standard for `aramwrap`.
* Added a FSL `Makefile` (dependent on the new `fsl/base` project).
* Re-arrangements to ease testing.


0.4.3 (Tuesday 22nd December 2020)
----------------------------------


* Fixed a bug in the ``NEWMAT::Matrix.SubMatrix`` method, which only seems
  to occur when compiling with `-std` newer than `c++98`.


0.4.2 (Thursday 17th December 2020)
-----------------------------------


* The ``armawrap`` ``NEWMAT::SVD`` function explicitly fails when given a
  matrix which has less rows than columns, in order to preserve compatibility
  with the behaviour of the original ``NEWMAT::SVD`` function.


0.4.1 (Thursday 26th September 2019)
------------------------------------


* Error messages originating from ``armadillo`` (e.g. ``chol(): failed to
  converge``) are no longer emitted.



0.4.0 (Wednesday 18th September 2019)
-------------------------------------


* Updated to ``armadillo`` 9.700.


0.3.1 (Wednesday 13th February 2019)
------------------------------------


* Fix to ensure that the results of expressions are formatted consistently
  when printed via ``std::cout``.


0.3.0 (Monday 14th January 2019)
--------------------------------


* ``newmat`` exception types are now typedefs of ``std::runtime_error``,
  rather than inheriting from ``armawrap::AWException``. This is so that
  existing code which is expecting a ``newmat`` exception is likely to catch
  exceptions raised by ``armadillo``.


0.2.4 (Friday 11th January 2019)
--------------------------------


* Improved the unit test for the ``Cholesky`` function.


0.2.3 (Friday 11th January 2019)
--------------------------------


* Fixed a bug in the ``Cholesky`` function - ``newmat`` returns a
  lower-triangular matrix, whereas ``armadillo`` defaults to returning an
  upper-triangular matrix.


0.2.2 (Monday 19th November 2018)
---------------------------------


* Inhibited class/struct alignment attributes in the ``armadillo`` code base,
  as they interfere with CUDA code.


0.2.1 (Friday 12th October 2018)
--------------------------------


* Adjustments to ``AWBase::operator<<`` operator overloads.


0.2.0 (Thursday 4th October 2018)
---------------------------------


* Fixes to ``.Storage`` calculation on certain subviews
* Refactored tests, and added a ``run_tests.sh`` script to make testing
  easier. Added GitLab CI integration for automated testing on pushes.
* Newmat source code is now included in the ``tests`` directory.


0.1.0 (Tuesday 2nd October 2018)
--------------------------------


* Fixed bug in subview lookup/assignment for non-simple matrix types
* Added ability to call ``.Storage`` on sub-views (except for band matrix
  types).


0.0.0
-----

* Initial working implementation
