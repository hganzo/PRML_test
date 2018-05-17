module MyModule
using Lib

using BigLib: thing1, thing2

import Base.show

importall OtherLib

export MyType, foo

struct MyType
    x
    end

    bar(x) = 2x
    foo(a::MyType) = bar(a.x) + 1

    show(io::IO, a::MyType) = print(io, "MyType $(a.x)")
end



